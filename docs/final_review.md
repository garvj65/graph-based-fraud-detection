# Final technical and code review

The required data, baseline, graph-feature, GNN, test-evaluation and monitoring
work is complete. README/About presentation remains deliberately deferred.
This review compares the implementation with the original project requirements,
checks the supported configuration, and preserves the historical experiment.

## Outcome

The strongest whole-period results come from XGBoost. Additional structure gives
a small AP improvement without a mean F1 improvement; the tested GNNs do not win
over the whole future period. Monitoring exposes a severe reliability failure
from the reported step-43 boundary. The selected model's pooled F1 changes from
0.8572 in steps 35–42 to 0.0492 in steps 43–49. Its post-event known-illicit recall
is 4.6% while the all-node alert rate remains 9.3%. Workload continues while
known-illicit detection deteriorates.

Those numbers come from `results/stage5/` and `results/stage6/`, with five-seed
means/sample std and explicit label supports. AP also depends on prevalence:
the labeled illicit rate changes from 9.16% to 2.53%. A ROC-AUC drop from 0.969
to 0.692 and sharply reduced recall provide evidence beyond AP alone. The study
does not identify a causal effect of the market shutdown.

## Code review scope and evidence

Reviewed paths include CSV/ID alignment, temporal masks and induced graphs;
training-only preprocessing and imbalance weights; forward-chained helper
scores; GNN loss masks, graph caching/direction, checkpoint selection and seed
handling; validation-only thresholds; frozen-model inference; sparse-label
metric/budget denominators; monitoring references, rolling counts and warning
rules; artifact persistence, reproduction commands, tests and CI.

- 127 tests pass, including real tiny synthetic fits and integration checks.
- Whole-repository Ruff passes, including the EDA notebook.
- The Stage 6 independent audit checks 525 per-step quality rows, 735 score and
  feature drift rows, five reloaded candidate models, training fingerprints,
  feature ranking, warning rules and seed aggregates.
- All 398 historical source/result files covered by the review baseline remain
  byte-identical. The README remains unchanged. No prior score, cutoff or model
  selection was replaced after observing test outcomes.
- Source-level controls keep unknown nodes in graph inference while excluding
  their labels from loss/quality metrics. No model/scaler fit occurs in Stage 6.

No blocking correctness defect was found in the supported pipeline configuration.
That finding does not establish deployment readiness or universal model quality.

## Findings and disposition

| Finding | Disposition | Practical meaning |
|---|---|---|
| Two notebook style violations; lint previously outside hosted CI | Fixed; CI now lints the whole repository | Formatting regressions are caught with the existing correctness tests. |
| Applying the frozen prefix-24 score helper to training rows would create in-sample neighbor scores | Avoided in monitoring by a dedicated structural-input builder, verified against original training fingerprints | The selected candidate's training reference does not use that helper; no old inference source was changed. |
| PSI > 0.2 also fires in every validation step/seed comparison | Disclosed, retained as a declared heuristic | Drift is a prompt to investigate; it cannot isolate the shutdown or replace observed quality. |
| Three high-gain aggregates correlate strongly with training time | Added training-only diagnostics; no feature removal or test retuning | The explicit no-time ablation does not establish removal of implicit temporal information. |
| Exact saved GNN score replay varies in the refreshed runtime | Quantified and retained as an open reproducibility limitation | Original results remain canonical; see the portability evidence below. |

## Exact-score replay and numerical portability

The original Stage 5 independent audit passed in its execution environment.
During this final review, its strict bitwise replay fails in the refreshed
runtime for GNN scores. The original audit was not relaxed or overwritten.
`results/final_review/strict_stage5_replay_failure.txt` preserves that failure.

A separate descriptive replay reloads all 100 frozen models and reports:

- 60 tabular model/seed combinations reproduce scores bitwise.
- 40 GNN combinations have small score differences; maximum absolute difference
  is 0.0000020862, and maximum AP difference is 0.0000010853.
- Zero frozen-threshold classifications change across all transactions/runs.
- Two all-node top-100 step queues have an internal ordering change; their
  membership and known-illicit counts do not change.
- Saved scores, published metrics, cutoffs and monitoring inputs are unchanged.

The new runtime records Torch/PyG versions, CPU capability and Torch build
metadata. A precise hardware/backend cause has not been established. Deterministic
algorithms and direct dependency pins do not prove cross-runtime bitwise identity.
`scripts/audit_replay_portability.py` reports differences without accepting them
through a tolerance override. This limitation matters when reproducing exact
rank/threshold ties; the observed differences do not change the project findings.

The Stage 6 structural XGBoost reference reloads exactly, and monitoring reuses
the original Stage 5 score arrays, so its independent audit passes unchanged.

## Remaining scientific and operating limits

- Approximately 77% of the dataset is unlabeled. The labeled cohort need not
  represent the full population; operational alert precision cannot be inferred.
- Topology and supplied forward-neighbor aggregates describe completed
  snapshots. Transaction-arrival availability and upstream normalization are
  not independently reconstructible from this release.
- Actual label delays and usable entity identities are absent. Live warning
  latency and entity-level temporal independence cannot be audited here.
- Validation is reused for candidate/checkpoint/cutoff selection; thresholds
  optimize validation F1, not a measured financial cost or staffing budget.
- Training-score drift references are in-sample. Pooled training distributions
  also mix different temporal regimes; validation drift contextualizes this.
- Rare illicit support makes some individual steps unstable: one test step has
  only two known illicit transactions. Three-step support gates reduce, but do
  not remove, that uncertainty.
- The declared small GNN architectures, bidirectional messages and tuning budget
  limit the conclusion. Five seeds measure fitting variation on one dataset,
  not temporal confidence intervals or proof that GNNs cannot help elsewhere.

The supported recommendation is an investigation/retraining review using recent
adjudicated labels, followed by a new chronological challenger evaluation and
budget-aware threshold selection. No retraining success, deployment, automated
promotion or causal event detection is claimed.

## Reproduction and next presentation pass

With prior artifacts available, run `make stage6` for tests, monitoring, its
independent audit and generated findings. A fresh checkout must reproduce prior
training/evaluation stages first. Direct tested versions are recorded in
`requirements-stage6-tested.txt`; these are not a transitive environment lock.
For the descriptive original-artifact portability check, run
`python scripts/audit_replay_portability.py`. Its output is explicitly distinct
from the original strict Stage 5 audit.

The next README should lead with the question, the whole-period comparison,
the reliability collapse and its business consequence, followed by method,
reproduction and limitations. It should use a coherent research narrative,
not concatenate stage reports. No README/About changes are included here.
