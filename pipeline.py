"""
Retrieval pipeline on TREC Robust04 (ad-hoc document ranking, MAP).

Steps:
  1. Select the best baseline (Dirichlet smoothing or BM25) with nested
     cross-validation on the labeled training queries.
  2. Tune RM3 pseudo-relevance feedback on top of the baseline, for a plain
     unigram query and for a Sequential Dependence Model (SDM) query, using the
     same nested cross-validation and grid for both.
  3. Tune a weighted Reciprocal Rank Fusion of the plain and SDM rankings.
  4. Write three TREC-format run files for the unlabeled test queries:
       run_1.res - fused ranking (plain RM3 + SDM RM3)
       run_2.res - plain unigram + RM3
       run_3.res - SDM + RM3

Requirements: Indri's IndriRunQuery binary (on PATH, or set INDRI_RUNQUERY_BIN),
an Indri index of Robust04, the queries file (XML), and the qrels file.
Data is not included in this repository. See README.md.
"""

import argparse
import os
import re
import shutil
import random
import time
import json
import subprocess
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
from collections import defaultdict


# If IndriRunQuery isn't on PATH, set this or export INDRI_RUNQUERY_BIN.
INDRI_RUNQUERY_BIN = os.environ.get("INDRI_RUNQUERY_BIN")

N_FOLDS = 5
FOLD_SEED = 42
TOP_N = 1000

# Hyperparameter grids. Kept intentionally narrow — informed by prior
# exploration (wide grids showed a flat MAP plateau for mu across 250-3000
# RM3 fbOrigWeight showed a clear, reproducible optimum near 0.15-0.2 after
# discovering it had previously been left at Indri's implicit default).
BASELINE_MU_GRID = [500, 1000]
BASELINE_BM25_K1_GRID = [1.2, 1.6, 2.0]
RM3_FB_DOCS_GRID = [15]
RM3_FB_TERMS_GRID = [50]
RM3_FB_ORIG_WEIGHT_GRID = [0.15, 0.2, 0.3]
FUSION_K_GRID = [10, 20, 40, 60, 100]
FUSION_W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def resolve_indri_binary():
    if INDRI_RUNQUERY_BIN and os.path.isfile(INDRI_RUNQUERY_BIN) and os.access(INDRI_RUNQUERY_BIN, os.X_OK):
        return INDRI_RUNQUERY_BIN
    found = shutil.which("IndriRunQuery")
    if found:
        return found
    return None


# ---------------------------------------------------------------
# Query reformulation strategies
# ---------------------------------------------------------------
def identity_transform(raw_query_text):
    return raw_query_text


def extract_terms(indri_query_text):
    match = re.search(r'#combine\(\s*(.*?)\s*\)', indri_query_text)
    inner = match.group(1) if match else indri_query_text
    return inner.split()


def sdm_transform(raw_query_text, w_unigram=0.85, w_ordered=0.10, w_unordered=0.05, uw_size=8):
    terms = extract_terms(raw_query_text)
    if len(terms) < 2:
        return f"#combine({' '.join(terms)})"
    unigram_part = "#combine(" + " ".join(terms) + ")"
    ordered_bigrams = " ".join(f"#od1({terms[i]} {terms[i + 1]})" for i in range(len(terms) - 1))
    unordered_bigrams = " ".join(f"#uw{uw_size}({terms[i]} {terms[i + 1]})" for i in range(len(terms) - 1))
    ordered_part = f"#combine({ordered_bigrams})"
    unordered_part = f"#combine({unordered_bigrams})"
    return (f"#weight({w_unigram} {unigram_part} "
            f"{w_ordered} {ordered_part} "
            f"{w_unordered} {unordered_part})")


# ---------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------
class IRPipeline:
    def __init__(self, index_path, queries_path, qrels_path):
        self.index_path = index_path
        self.queries_path = queries_path
        self.qrels_path = qrels_path

        self.indri_bin = resolve_indri_binary()
        if self.indri_bin is None:
            raise FileNotFoundError(
                "IndriRunQuery not found on PATH and no valid INDRI_RUNQUERY_BIN override set.\n"
                "Run: find / -iname 'IndriRunQuery' 2>/dev/null\n"
                "Then: export PATH=$PATH:/path/to/indri/bin"
            )
        print(f"[Init] Using Indri binary: {self.indri_bin}")

        self.qrels = self._parse_qrels()
        self.queries = self._parse_queries()

        self.train_ids = sorted(set(self.queries.keys()) & set(self.qrels.keys()))
        self.test_ids = sorted(set(self.queries.keys()) - set(self.qrels.keys()))

        print(f"[Init] Total collection queries parsed: {len(self.queries)}")
        print(f"[Init] Labeled training queries isolated: {len(self.train_ids)}")
        print(f"[Init] Unjudged test queries isolated for final submission: {len(self.test_ids)}")

        if len(self.train_ids) != 50:
            print(f"[Warning] Expected 50 training queries, found {len(self.train_ids)}.")
        if len(self.test_ids) != 199:
            print(f"[Warning] Expected 199 test queries, found {len(self.test_ids)}.")

    def _parse_queries(self):
        queries = {}
        if not os.path.exists(self.queries_path):
            raise FileNotFoundError(f"Queries file not found: {self.queries_path}")
        tree = ET.parse(self.queries_path)
        root = tree.getroot()
        for query_node in root.findall('query'):
            num_node = query_node.find('number')
            text_node = query_node.find('text')
            if num_node is not None and text_node is not None:
                queries[num_node.text.strip()] = text_node.text.strip()
        return queries

    def _parse_qrels(self):
        qrels = defaultdict(set)
        if not os.path.exists(self.qrels_path):
            raise FileNotFoundError(f"Qrels file not found: {self.qrels_path}")
        with open(self.qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    q_id, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
                    if int(rel) > 0:
                        qrels[q_id].add(doc_id)
        return qrels

    def calculate_ap(self, ranked_list, rel_set):
        if not rel_set or not ranked_list:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in rel_set:
                hits += 1
                sum_precisions += hits / rank
        return sum_precisions / len(rel_set)

    def get_folds(self):
        ids = self.train_ids.copy()
        random.Random(FOLD_SEED).shuffle(ids)
        fold_size = len(ids) // N_FOLDS
        folds = [ids[i * fold_size:(i + 1) * fold_size] for i in range(N_FOLDS)]
        if len(ids) % N_FOLDS != 0:
            folds[-1].extend(ids[N_FOLDS * fold_size:])
        return folds

    def build_param_file(self, filename, query_dict, method="dirichlet", param_val=2500,
                          fb_docs=0, fb_terms=0, fb_orig_weight=0.5, run_tag="run",
                          query_transform=identity_transform):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("<parameter>\n")
            f.write(f"  <index>{self.index_path}</index>\n")
            f.write("  <count>1000</count>\n")
            f.write("  <trecFormat>true</trecFormat>\n")
            f.write(f"  <runID>{run_tag}</runID>\n")

            if method == "dirichlet":
                f.write(f"  <rule>method:dirichlet,mu:{param_val}</rule>\n")
            elif method == "bm25":
                f.write(f"  <rule>method:okapi,k1:{param_val},b:0.75</rule>\n")
            else:
                raise ValueError(f"Unknown method: {method}")

            if fb_docs > 0 and fb_terms > 0:
                f.write(f"  <fbDocs>{fb_docs}</fbDocs>\n")
                f.write(f"  <fbTerms>{fb_terms}</fbTerms>\n")
                f.write(f"  <fbOrigWeight>{fb_orig_weight}</fbOrigWeight>\n")

            for q_id, q_text in query_dict.items():
                transformed_text = query_transform(q_text)
                escaped_text = saxutils.escape(transformed_text)
                f.write("  <query>\n")
                f.write(f"    <number>{q_id}</number>\n")
                f.write(f"    <text>{escaped_text}</text>\n")
                f.write("  </query>\n")
            f.write("</parameter>\n")

    def run_indri_safely(self, param_file, output_res_file):
        try:
            with open(output_res_file, 'w') as out_f:
                result = subprocess.run(
                    [self.indri_bin, param_file],
                    stdout=out_f, stderr=subprocess.PIPE,
                    text=True, check=False,
                )
            if result.returncode != 0:
                print(f"[Process Warning] IndriRunQuery failed: {result.stderr.strip()[:300]}")
                return False
            return True
        except Exception as e:
            print(f"[Process Exception] {e}")
            return False

    def parse_results(self, res_file, warn_on_violation=False):
        results = defaultdict(list)
        if not os.path.exists(res_file):
            return results
        last_scores = defaultdict(lambda: float('inf'))
        with open(res_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    q_id, doc_id, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
                    if warn_on_violation and score > last_scores[q_id]:
                        print(f"[Format Violation] Non-descending score for query {q_id}")
                    last_scores[q_id] = score
                    results[q_id].append((doc_id, rank, score))
        return results


# ---------------------------------------------------------------
# Generic nested cross-validation selector
# ---------------------------------------------------------------
def nested_cv_select(per_config_query_ap, folds, all_ids, label=""):
    valid_configs = [c for c in per_config_query_ap if per_config_query_ap[c]]
    if not valid_configs:
        print(f"[{label}] All configurations failed.")
        return None, 0.0, []

    outer_fold_scores = []
    selected_per_fold = []
    for fold_idx in range(N_FOLDS):
        val_set = set(folds[fold_idx])
        train_set = [q for q in all_ids if q not in val_set]

        best_config, best_train_avg = None, -1.0
        for config in valid_configs:
            train_avg = sum(per_config_query_ap[config][q] for q in train_set) / len(train_set)
            if train_avg > best_train_avg:
                best_train_avg = train_avg
                best_config = config

        val_avg = sum(per_config_query_ap[best_config][q] for q in val_set) / len(val_set)
        outer_fold_scores.append(val_avg)
        selected_per_fold.append(best_config)
        print(f"[{label} Nested CV] fold={fold_idx} selected={best_config} -> held-out MAP={val_avg:.4f}")

    nested_cv_map = sum(outer_fold_scores) / len(outer_fold_scores)

    final_config, final_avg = None, -1.0
    for config in valid_configs:
        full_avg = sum(per_config_query_ap[config][q] for q in all_ids) / len(all_ids)
        if full_avg > final_avg:
            final_avg = full_avg
            final_config = config

    print(f"[{label} Summary] nested_cv_map(honest, held-out)={nested_cv_map:.4f}  "
          f"production_config={final_config} (full-query avg MAP={final_avg:.4f}) "
          f"per-fold selections={selected_per_fold}")

    return final_config, nested_cv_map, selected_per_fold


# ---------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------
def get_per_query_ap(pipeline, method, param_val, fb_docs, fb_terms, fb_orig_weight,
                      query_transform, tag):
    all_queries = {q: pipeline.queries[q] for q in pipeline.train_ids}
    param_file = f"tmp_{tag}.xml"
    res_file = f"tmp_{tag}.res"
    try:
        pipeline.build_param_file(param_file, all_queries, method=method, param_val=param_val,
                                   fb_docs=fb_docs, fb_terms=fb_terms, fb_orig_weight=fb_orig_weight,
                                   run_tag=tag, query_transform=query_transform)
        success = pipeline.run_indri_safely(param_file, res_file)
        if not success:
            return {}
        retrieved = pipeline.parse_results(res_file)
        if not any(retrieved.values()):
            return {}
        return {q: pipeline.calculate_ap([d for d, r, s in retrieved[q]], pipeline.qrels[q])
                for q in pipeline.train_ids}
    finally:
        if os.path.exists(param_file):
            os.remove(param_file)
        if os.path.exists(res_file):
            os.remove(res_file)


def get_scored_ranking(pipeline, query_dict, method, param_val, fb_docs, fb_terms,
                        fb_orig_weight, query_transform, tag):
    param_file = f"tmp_{tag}.xml"
    res_file = f"tmp_{tag}.res"
    try:
        pipeline.build_param_file(param_file, query_dict, method=method, param_val=param_val,
                                   fb_docs=fb_docs, fb_terms=fb_terms, fb_orig_weight=fb_orig_weight,
                                   run_tag=tag, query_transform=query_transform)
        success = pipeline.run_indri_safely(param_file, res_file)
        if not success:
            return {}
        return pipeline.parse_results(res_file, warn_on_violation=True)
    finally:
        if os.path.exists(param_file):
            os.remove(param_file)
        if os.path.exists(res_file):
            os.remove(res_file)


def weighted_rrf(list_a, list_b, k, w):
    rank_a = {d: r for d, r, s in list_a}
    rank_b = {d: r for d, r, s in list_b}
    doc_ids = set(rank_a.keys()) | set(rank_b.keys())
    scores = {}
    for d in doc_ids:
        ra = rank_a.get(d, 100000)
        rb = rank_b.get(d, 100000)
        scores[d] = w * (1.0 / (k + ra)) + (1.0 - w) * (1.0 / (k + rb))
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------
# Submission writing / validation
# ---------------------------------------------------------------
def write_run_file(filename, retrieved, all_ids, run_tag, top_n=TOP_N):
    with open(filename, 'w') as out_f:
        n_written = 0
        for q in all_ids:
            entries = sorted(retrieved.get(q, []), key=lambda x: x[1])[:top_n]
            if not entries:
                continue
            for doc_id, rank, score in entries:
                out_f.write(f"{q} Q0 {doc_id} {rank} {score:.8f} {run_tag}\n")
            n_written += 1
    return n_written


def validate_run_file(filename, expected_queries):
    with open(filename) as f:
        lines = f.readlines()
    print(f"[Validation] {filename}: {len(lines)} total lines.")
    last_score_by_q = {}
    violations = 0
    seen_qids = set()
    for line in lines:
        parts = line.strip().split()
        qid, score = parts[0], float(parts[4])
        seen_qids.add(qid)
        if qid in last_score_by_q and score > last_score_by_q[qid]:
            violations += 1
        last_score_by_q[qid] = score
    print(f"[Validation] {filename}: {len(seen_qids)}/{expected_queries} unique queries, "
          f"{violations} non-descending score violations.")


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Robust04 retrieval pipeline (Indri).")
    parser.add_argument("--index", required=True, help="Path to the Indri index of Robust04")
    parser.add_argument("--queries", required=True, help="Path to the queries XML file")
    parser.add_argument("--qrels", required=True, help="Path to the qrels file for the training queries")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    pipeline = IRPipeline(args.index, args.queries, args.qrels)
    folds = pipeline.get_folds()

    # --- Phase 1: baseline retrieval model selection ---
    print("\n--- Phase 1: Baseline Model Selection (Dirichlet vs BM25) ---")
    dirichlet_ap = {mu: get_per_query_ap(pipeline, "dirichlet", mu, 0, 0, 0.5,
                                          identity_transform, f"base_mu{mu}")
                     for mu in BASELINE_MU_GRID}
    best_mu, dirichlet_map, _ = nested_cv_select(dirichlet_ap, folds, pipeline.train_ids, "Dirichlet")

    bm25_ap = {k1: get_per_query_ap(pipeline, "bm25", k1, 0, 0, 0.5,
                                     identity_transform, f"base_k1{k1}")
               for k1 in BASELINE_BM25_K1_GRID}
    best_k1, bm25_map, _ = nested_cv_select(bm25_ap, folds, pipeline.train_ids, "BM25")

    if best_mu is not None and (best_k1 is None or dirichlet_map >= bm25_map):
        baseline_method, baseline_param = "dirichlet", best_mu
    else:
        baseline_method, baseline_param = "bm25", best_k1
    print(f"\n[Decision] Baseline: {baseline_method}, param={baseline_param} "
          f"(MAP={max(dirichlet_map, bm25_map):.4f})")

    # --- Phase 2: RM3 tuning, plain unigram formulation ---
    print("\n--- Phase 2a: RM3 Tuning — Plain Unigram Formulation ---")
    plain_rm3_ap = {}
    for docs in RM3_FB_DOCS_GRID:
        for terms in RM3_FB_TERMS_GRID:
            for orig_w in RM3_FB_ORIG_WEIGHT_GRID:
                config = (docs, terms, orig_w)
                plain_rm3_ap[config] = get_per_query_ap(
                    pipeline, baseline_method, baseline_param, docs, terms, orig_w,
                    identity_transform, f"plain_rm3_{docs}_{terms}_{orig_w}")

    plain_config, plain_rm3_map, _ = nested_cv_select(plain_rm3_ap, folds, pipeline.train_ids, "Plain-RM3")
    plain_docs, plain_terms, plain_orig_w = plain_config

    # --- Phase 2b: RM3 tuning, SDM proximity formulation ---
    print("\n--- Phase 2b: RM3 Tuning — SDM Proximity Formulation ---")
    sdm_rm3_ap = {}
    for docs in RM3_FB_DOCS_GRID:
        for terms in RM3_FB_TERMS_GRID:
            for orig_w in RM3_FB_ORIG_WEIGHT_GRID:
                config = (docs, terms, orig_w)
                sdm_rm3_ap[config] = get_per_query_ap(
                    pipeline, baseline_method, baseline_param, docs, terms, orig_w,
                    sdm_transform, f"sdm_rm3_{docs}_{terms}_{orig_w}")

    sdm_config, sdm_rm3_map, _ = nested_cv_select(sdm_rm3_ap, folds, pipeline.train_ids, "SDM-RM3")
    sdm_docs, sdm_terms, sdm_orig_w = sdm_config

    # --- Phase 3: fusion tuning ---
    print("\n--- Phase 3: Fusion Tuning (RRF over Plain-RM3 and SDM-RM3) ---")
    all_train_queries = {q: pipeline.queries[q] for q in pipeline.train_ids}
    retrieved_plain = get_scored_ranking(pipeline, all_train_queries, baseline_method, baseline_param,
                                          plain_docs, plain_terms, plain_orig_w,
                                          identity_transform, "fusion_plain_train")
    retrieved_sdm = get_scored_ranking(pipeline, all_train_queries, baseline_method, baseline_param,
                                        sdm_docs, sdm_terms, sdm_orig_w,
                                        sdm_transform, "fusion_sdm_train")

    fusion_ap = {}
    for k in FUSION_K_GRID:
        for w in FUSION_W_GRID:
            fused_ranked = {q: [d for d, s in weighted_rrf(retrieved_plain.get(q, []),
                                                             retrieved_sdm.get(q, []), k, w)]
                             for q in pipeline.train_ids}
            fusion_ap[(k, w)] = {q: pipeline.calculate_ap(fused_ranked[q], pipeline.qrels[q])
                                  for q in pipeline.train_ids}

    fusion_config, fusion_map, _ = nested_cv_select(fusion_ap, folds, pipeline.train_ids, "Fusion")
    best_k, best_w = fusion_config

    # --- Phase 4: final submission runs on the 199 test queries ---
    print("\n--- Phase 4: Final Submission Runs (199 test queries) ---")
    test_queries = {q: pipeline.queries[q] for q in pipeline.test_ids}
    assert len(test_queries) == len(pipeline.test_ids), "Test query subset size mismatch."

    print("Retrieving plain unigram + RM3 for test queries...")
    test_plain = get_scored_ranking(pipeline, test_queries, baseline_method, baseline_param,
                                     plain_docs, plain_terms, plain_orig_w,
                                     identity_transform, "final_plain_test")

    print("Retrieving SDM + RM3 for test queries...")
    test_sdm = get_scored_ranking(pipeline, test_queries, baseline_method, baseline_param,
                                   sdm_docs, sdm_terms, sdm_orig_w,
                                   sdm_transform, "final_sdm_test")

    fused_test = {}
    for q in pipeline.test_ids:
        list_a = test_plain.get(q, [])
        list_b = test_sdm.get(q, [])
        if not list_a and not list_b:
            continue
        fused = weighted_rrf(list_a, list_b, best_k, best_w)[:TOP_N]
        fused_test[q] = [(doc_id, rank, score) for rank, (doc_id, score) in enumerate(fused, start=1)]

    n1 = write_run_file("run_1.res", fused_test, pipeline.test_ids, "run_1")
    n2 = write_run_file("run_2.res", test_plain, pipeline.test_ids, "run_2")
    n3 = write_run_file("run_3.res", test_sdm, pipeline.test_ids, "run_3")

    print(f"\n[Summary] run_1 (fusion): {n1}/{len(pipeline.test_ids)} queries written")
    print(f"[Summary] run_2 (plain + RM3): {n2}/{len(pipeline.test_ids)} queries written")
    print(f"[Summary] run_3 (SDM + RM3): {n3}/{len(pipeline.test_ids)} queries written")

    validate_run_file("run_1.res", len(pipeline.test_ids))
    validate_run_file("run_2.res", len(pipeline.test_ids))
    validate_run_file("run_3.res", len(pipeline.test_ids))

    summary = {
        "baseline_method": baseline_method,
        "baseline_param": baseline_param,
        "baseline_cv_map": max(dirichlet_map, bm25_map),
        "run_2_plain_rm3": {"fbDocs": plain_docs, "fbTerms": plain_terms,
                             "fbOrigWeight": plain_orig_w, "nested_cv_map": plain_rm3_map},
        "run_3_sdm_rm3": {"fbDocs": sdm_docs, "fbTerms": sdm_terms,
                           "fbOrigWeight": sdm_orig_w, "nested_cv_map": sdm_rm3_map},
        "run_1_fusion": {"rrf_k": best_k, "rrf_w": best_w, "nested_cv_map": fusion_map},
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open("final_submission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Total elapsed time: {summary['elapsed_seconds']:.1f}s")
    print("[Done] Summary written to final_submission_summary.json")


if __name__ == "__main__":
    main()