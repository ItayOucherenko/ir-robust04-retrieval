# Robust04 Retrieval: Dirichlet/BM25 + RM3 + SDM + RRF

Ad-hoc document retrieval on TREC Robust04 (news collection), built on Indri. The pipeline selects its settings with nested cross-validation, then writes three TREC-format run files.

Built together with [Amit Gertner](https://github.com/amit22882036-ship-it).

## Pipeline

1. **Baseline:** Dirichlet-smoothed query likelihood (mu in {500, 1000}) or BM25 (k1 in {1.2, 1.6, 2.0}), chosen by nested CV.
2. **RM3 pseudo-relevance feedback:** top 15 documents, 50 expansion terms, original-query weight in {0.15, 0.2, 0.3}.
3. **Two query formulations for RM3:** a plain unigram query, and a Sequential Dependence Model (SDM) query (unigrams 0.85, ordered windows 0.10, unordered windows of size 8 at 0.05).
4. **Fusion:** weighted Reciprocal Rank Fusion of the plain and SDM rankings (rrf k in {10, 20, 40, 60, 100}, weight in {0.3 ... 0.7}).
5. **Output:** `run_1.res` (fusion), `run_2.res` (plain + RM3), `run_3.res` (SDM + RM3).

## Results (MAP, 50 labeled training queries)

| Stage | MAP |
| --- | --- |
| Baseline (Dirichlet or BM25) | 0.230 |
| + RM3, plain unigram | 0.257 |
| + RM3, SDM | 0.258 |
| Fusion of plain and SDM (selected run) | 0.258 |

RM3 gives most of the gain. SDM and the fusion are comparable to plain RM3 in this setup; they do not beat it by a measurable margin.

**Caveats:**
- The grid was narrowed after looking at results on the same 50 queries. The nested CV is correct inside the procedure, but the reported MAP is somewhat optimistic for unseen queries.
- The plain and SDM configurations fed into the fusion stage were selected on all 50 training queries, so the fusion's nested-CV number has mild leakage.

## Requirements

- Python 3
- Indri with the `IndriRunQuery` binary, on `PATH` or set in `INDRI_RUNQUERY_BIN`
- An Indri index of TREC Robust04, the Robust04 queries file (XML), and the qrels file for the training queries

The data is not included in this repository. Robust04 is distributed by NIST.

## Usage

```bash
export INDRI_RUNQUERY_BIN=/path/to/indri/bin/IndriRunQuery   # if not on PATH
python3 pipeline.py --index /path/to/ROBUSTindex --queries /path/to/queries.xml --qrels /path/to/qrels
```

Outputs: `run_1.res`, `run_2.res`, `run_3.res` (TREC format) and `final_submission_summary.json` in the working directory.
