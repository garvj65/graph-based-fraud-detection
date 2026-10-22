# Stage 5: final evaluation protocol

Stage 5 evaluates frozen models. It does not train or tune them. README, About,
repository presentation and Stage 6 monitoring are deferred until their review
points, as requested.

## What is frozen before test

- Train 1–24, validation 25–34, test 35–49. No random split or cross-time shuffle.
- Every selected Stage 2–4 fit: 100 model/seed combinations across 20 variants.
- Five seeds per variant. Headline results cover seven primary variants:
  LogReg all, XGBoost local/all, XGBoost with structure and with structure plus
  neighbor scores, GCN all, and GraphSAGE all.
- Original per-run validation-max-F1 cutoffs, with `score >= threshold`.
- Average precision as primary PR-AUC; trapezoidal PR-AUC is separate and
  ROC-AUC is secondary. Unknown labels never become negatives.
- Completed-snapshot availability and the previously declared graph definitions.
- Highest mean validation AP among headline variants chooses the candidate for
  subsequent monitoring. Exact ties keep the declared variant order. Test
  ranking does not replace that selection, and representative seed 0 is fixed.

The evaluator first checks hashes of weights, scalers, score helper and input
files. It replays all original validation predictions with an absolute
tolerance of 1e-12, recording whether replay is bitwise identical. Frozen
confusion counts are also compared and any differences are recorded, not
corrected by changing thresholds. Minor floating-point differences can arise
when a linear model scores all nodes rather than only labeled rows.

The protocol is persisted, hashed and marked frozen before any test prediction.
It includes source-record hashes, raw/source hashes, effective configuration,
validation replay checks, model selection and auxiliary recall policies.
Scoring rejects a changed source record or primary threshold. No training API
is called. Unit tests explicitly replace estimator/scaler fitting with errors.

## Alert budgets and sparse labels

The choice form returned no selections. Before test scoring, the explicit
benchmark default was set to **up to 100 alerts per time step**, with both views
reported. This is a comparison policy, not a claim about an actual fraud team's
capacity. Ties are broken by lexical transaction ID, independent of labels.

1. **Labeled-only precision@100:** filter to known labels, rank, then take up to
   100 per time step. This is retrospective and cannot represent a live queue
   when correctness is unknown at scoring time. Overall precision is computed
   from total known illicit alerts divided by total selected labeled alerts,
   rather than an unweighted average of step precisions.
2. **All-transaction queue:** rank all observed transactions, including unknowns.
   Report known illicit, known licit, unknown and total alerts, plus label
   coverage. Precision among the known-label subset is clearly named and is
   not overall operational precision. Unknown alerts have no assigned truth.

The fixed-k policy is separate from the primary frozen-threshold policy. A
queue can include scores below the F1 threshold. Budgets are capped by cohort
size and the actual count is recorded. Test totals are averaged over seeds;
repeated seeds do not create additional distinct transactions.

## Recall and workload diagnostics

Before test, additional cutoffs are selected on validation to meet predeclared
known-illicit recall targets of 50%, 80% and 90%. The highest cutoff meeting a
target is used, including score ties as a group. These do not replace the
primary F1 cutoffs. The exact cutoffs are frozen and transferred to test;
achieved test recall is allowed to miss the original target.

The report includes validation PR tradeoffs, all-node investigation workload,
and transferred-policy outcomes. It does not select a test-optimal threshold
or imply an actual monetary false-positive/false-negative cost.

## Graph inference and leakage boundaries

Test structure is computed independently within each observed snapshot 35–49.
Degree, directed PageRank and clustering reuse Stage 3's definitions. This is
an inference transform on available topology, not fitting a global graph or
normalizer across test periods. Future snapshots cannot affect earlier ones.

Neighbor scores use the saved prefix-24 helper and its training-only scaler.
There is no new helper fit and no raw neighbor-label feature. GNN inference
uses saved scalers, unique bidirectional edges within snapshots, no cached
training adjacency, and evaluation mode without dropout. Unknown nodes stay
in messages but are removed from all classification metrics. The inference
interface accepts observations without any label vector.

Supplied aggregates include forward-neighbor information, and upstream
standardization/as-of availability cannot be independently reconstructed.
Completed snapshots are retrospective and do not prove transaction-arrival
availability. The observed labeled cohort can be unrepresentative, and
label-availability times are missing. These limitations persist after a
successful temporal backtest.

## Reproduction and verification

Use the tested CPU Torch/PyG stack from Stage 4. In a checkout with the prior
model artifacts available:

```bash
make stage5
# Or:
python -m src.final_evaluate --config configs/stage5.yaml
python scripts/validate_stage5.py
```

Model artifacts are intentionally excluded from Git. A fresh checkout must
reproduce the earlier training stages before evaluating the models from those
new manifests; Stage 5 refuses to silently refit missing or changed models.
The recorded experiment uses the exact original Stage 2–4 artifacts. Historical
result numbers are not reused as newly measured predictions.

Results include the primary comparison, all feature/message/time ablations,
per-seed metrics and confusion matrices, both budget denominators, validation
recall policies transferred to test, protocol/feature audits, three figures,
and representative score/queue CSVs. Full scores for every model, including
unknown nodes, remain under ignored `artifacts/` for Stage 6 reuse.

The independent validator reloads every model, reproduces test scores, checks
unchanged thresholds, independently computes AP/confusion counts, re-sorts
budget queues using a separate implementation, reconstructs recall cutoffs
from validation only, and verifies means/std over the exact five seeds.

## Artifact recovery in the recorded run

The first evaluation completed all 100 model/seed predictions and wrote their
metric table, but report generation exposed incomplete score archives and
progress logs. Ten saves were repaired from the same frozen protocol; 90 valid
evaluation records were reused. The original 100-row metric table and failed
batch record are retained under `results/stage5/recovery_evidence/`.

JSON/NPZ serialization was changed to atomic replacement with file/directory
fsync. The original frozen protocol remains unchanged. Its original evaluation
source is retained, and an AST comparison verifies that only imports and the
NPZ-writer alias changed: model inference, metrics, splits, budget, checkpoint
choice and threshold logic are identical. Every original metric was reproduced.
`artifact_recovery.json` records old/new source hashes and the repair ledger.
The independent validator checks this narrow amendment and reloads every model.
New runs use atomic writes from the start. This was a persistence repair, not
an opportunity to tune after observing test outcomes.

No per-step performance monitoring, drift threshold, retraining trigger or
shutdown causality is established in Stage 5. Those remain Stage 6.
