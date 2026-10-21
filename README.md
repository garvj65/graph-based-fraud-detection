# Graph-based illicit transaction detection on Elliptic

**Question:** Do additional graph features and small GNNs improve on a strong
tabular baseline, and does the model remain reliable over time?

**Status: stage 1 complete; paused for review.** Real data has been downloaded,
audited and split. No classifier has been trained. Model comparisons, monitoring
performance, lift, seed variance and interview conclusions about model quality
are deliberately absent until measured in the later stages.

This is an offline benchmark for an Associate Data Scientist portfolio. It is
not a real-time fraud service or a claim of deployment experience.

## Reproduce stage 1

Python 3.10+; the completed run used Python 3.12.14. No GPU is required.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make stage1
```

`make stage1` runs the tests, downloads the three raw CSVs from the public PyG
mirror if needed, validates the full dataset, and regenerates the audit and EDA.
On Windows, use `.venv\Scripts\activate` and run these commands instead of Make:

```bash
python -m pytest -q
python -m src.data --config configs/default.yaml --download
```

The raw feature CSV is about 690 MB; allow space for raw files, temporary ZIPs
and parsing memory. Raw data is excluded from version control and this archive.
The CSV layout is the same as the original Kaggle release. Kaggle downloads can
be supplied with:

```bash
python -m src.data --raw-dir /path/to/elliptic_bitcoin_dataset
```

The requested PyG acquisition route is also implemented:

```bash
python -m pip install -r requirements-gnn.txt
python -m src.data --source pyg
```

PyG acquisition was not exercised in the current environment (Torch/PyG are not
installed); the completed full-data run and tests use raw CSV loading. PyG's
processed feature matrix omits time, and its default training mask uses steps
1–34. This project ignores those masks, recovers time from the raw CSV and joins
labels by transaction ID, rather than trusting CSV row-order alignment.

`requirements-stage1-tested.txt` records the direct package versions used for
the completed run on Python 3.12; it is not a cross-platform or transitive lock.
GitHub Actions is configured for Python 3.10 and 3.12, using small synthetic test
fixtures without a network/data download. Consult the repository's
[Actions tab](https://github.com/garvj65/graph-based-fraud-detection/actions)
for hosted CI status; the local verification is saved with this checkpoint.

## Fixed temporal protocol

Inclusive boundaries are enforced in code and configuration. There is no random
split and no temporal shuffling. Counts below come from the completed raw-data
audit at `results/stage1/data_audit.json`.

| Split | Time steps | All nodes | Labeled nodes | Illicit | Licit | Unknown |
|---|---|---:|---:|---:|---:|---:|
| Train | 1–24 | 108,950 | 23,606 | 2,219 | 21,387 | 85,344 |
| Validation | 25–34 | 27,315 | 6,288 | 1,243 | 5,045 | 21,027 |
| Test | 35–49 | 67,504 | 16,670 | 1,083 | 15,587 | 50,834 |

Internal labels are `1=illicit`, `0=licit`, `-1=unknown`. The raw CSV uses
`1=illicit`, `2=licit`, `unknown=unlabeled`. Unknown nodes remain in their
available graph, but never enter supervised masks. Future-period unknown nodes
are not available to the training graph.

The audit found 203,769 nodes, 234,355 directed edges, 49 time steps, zero
cross-time edges, zero duplicate directed edges and zero self-loops. The
training graph retains all 108,950 training-period nodes and 127,488 edges.

### The 166-feature counting detail

The published 94 local features **include the time step**. The CSV therefore
contains ID, time, 93 local numeric features, and 72 aggregate numeric features.
PyG returns 165 predictors. The project stores time separately and supports both
93/165-feature and 94/166-feature matrices, with an explicit `include_time`
argument. The primary predictor policy remains undecided pending review.

The all-feature tree baseline already includes supplied graph-derived neighbor
aggregates. It is a tabular model, but cannot be described as having no graph
information. Later comparisons must distinguish supplied aggregates, newly
computed graph features, and learned message passing.

## Stage 1 EDA

![Class imbalance, label coverage, observed illicit prevalence and degree distribution](results/stage1/figures/data_overview.png)

The observed illicit rate uses **illicit / known labels**. It is not the
population fraud rate: unknown nodes have no negative label. The CSV includes
label coverage and the known-illicit share of all nodes under separate names.
Full-period plots are descriptive and must not be used to tune models or
thresholds. The test period remains unused for model selection.

Degree in this figure means in-degree + out-degree in the raw directed graph.
It is descriptive EDA, not a feature fitted on the complete graph for training.
The notebook is a thin view over the same code and saved evidence, with no
independent preprocessing or split implementation.

## Leakage risks and defenses

| Risk | Defense / remaining limitation |
|---|---|
| PyG train mask accidentally includes validation | Rebuild all masks from raw timestamps; reject any departure from 1–24 / 25–34 / 35–49. |
| Labels silently attached to the wrong transactions | Join by ID; reject duplicate/missing IDs, invalid classes and missing edge endpoints. |
| Unknown interpreted as licit | Use `-1` and separate graph/supervision masks; test every split. |
| Future nodes affect training messages or normalization | Training graph is induced on steps 1–24; snapshot API exposes one time step only. Unknown future nodes are excluded from training too. |
| Neighbor-label leakage | Helper exposes train labels only; validation/test labels are replaced by unknown. Training label aggregates will still require cross-fitting or a defensible label-availability protocol in stage 3. |
| Label-derived features fail outside training | There are no cross-time edges. The audit finds zero evaluation edges touching train-labeled nodes. A train-label-only neighbor feature has no labeled neighbors in validation/test; do not claim useful holdout coverage. |
| Supplied aggregates use forward neighbors | These are retrospective one-hop aggregates. Completed-snapshot versus transaction-arrival availability must be agreed before modeling. The anonymized release cannot certify strict as-of feature availability or reconstruct precise arrival order. |
| Preprocessing fitted on validation/test | None is fitted in stage 1. Later scalers, class weights, feature selection and drift reference/bin edges must use training data only. |
| Operational precision overstated despite sparse labels | Keep label coverage visible; unknown alerts cannot be assigned correctness. Labeled-only top-k precision and an all-node alert queue answer different questions. |
| Test-guided choices | Use validation only for early stopping, hyperparameters, threshold and model selection. Freeze them before final test evaluation. |

PageRank, clustering and other structural values on a new observed snapshot
are inference-time computations, not permission to estimate a global graph
using future steps. Stage 3 must document this distinction for every feature.

## Evidence and reproducibility

Each stage-1 invocation writes a run JSON with its effective configuration,
package versions, raw-file SHA-256 hashes, source-file hashes, audit output and
status. Failures after config parsing also leave a failure record. Seed is
`null` because stage 1 has no stochastic model fitting; seeds 0–4 are reserved
in config for later experiments. No placeholder model metric is emitted.

Use `results/stage1/latest_run.json` for the current successful run. One early
audit run is explicitly marked `invalidated`: its duplicate-edge counter used
the wrong array dimension. The bug was fixed, a regression assertion added,
and the full-data audit rerun. Its numbers are not current results.

Outputs:

- `results/stage1/data_audit.json`: split, class and graph integrity checks.
- `results/stage1/split_summary.csv`: all-node and labeled counts per split.
- `results/stage1/label_coverage_by_step.csv`: descriptive counts and rates.
- `results/stage1/figures/data_overview.png`: one four-panel EDA figure.
- `results/stage1/runs/`: per-run evidence, including failure/invalidation status.
- `results/stage1/verification.txt`: completed local test/lint verification.
- `docs/stage1_findings.md`: five interview-style data/protocol findings.

Tests cover exact boundaries and ordering, unknown-label exclusion, row-order
independence, future-input invariance, graph membership, invalid schema/edges,
rate denominators, and successful/failed CLI logging. They use explicitly
synthetic fixtures; dataset findings use only the downloaded real release.

## Remaining review stages

1. **Data + splits + tests:** complete, awaiting review.
2. **Baselines:** weighted logistic regression and XGBoost on local/all inputs;
   train-only preprocessing and five-seed run logs.
3. **Graph features:** NetworkX degree, PageRank, clustering and defensible
   neighbor statistics; evaluate incremental validation lift.
4. **GNNs:** small GCN then GraphSAGE; weighted supervised loss, training-period
   message passing, validation early stopping, five seeds.
5. **Evaluation:** fixed validation-selected thresholds; AP/PR-AUC definition to
   confirm; precision/recall/F1, confusion matrix, top-k and secondary ROC-AUC;
   mean ± sample std, paired seed differences and feature/structure ablations.
6. **Monitoring:** frozen-model per-step test metrics, alert rates and score
   distributions; train-reference PSI/KS and top-10 feature drift. Investigate
   the reported step-43 regime change without asserting it causes measured
   decay. Propose triggers with training/validation justification and label-delay
   assumptions. Emit plots, JSON/CSV and five final interview findings.

Stages 2–6 are not implemented in this checkpoint. No model results table or
monitoring claims will be added before the corresponding runs. Explainability,
communities, semi-supervised experiments and a dashboard follow the core
comparison if time permits.

## Decisions pending before modeling

The config retains `null` for decisions that have not been answered:

- Include time as a predictor? Proposed main run: exclude it (93/165 numeric
  predictors); optionally compare 94/166 as a labeled ablation.
- Information availability? Proposed main run: completed snapshots, with
  forward-neighbor and label-availability caveats.
- PR metric definition? Proposed: sklearn average precision, labeled
  **AP (PR-AUC)** consistently, rather than silently mixing integration methods.
- Threshold objective? Choose validation F1 or a precision/alert-budget policy;
  agree on tie breaking and freeze before test.
- Top-k budget and universe? Define an alert budget per step and distinguish
  labeled-only evaluation from all-node operational review volume.

## Limitations

Most labels are unknown, and observed labels need not be representative or
available promptly. Label availability times are absent. Ground truth is an
entity-derived heuristic classification, not a proof that a transaction is a
crime. Features are anonymized and partly retrospective, constraining
explanation and strict causal replay. Temporal label/prevalence shift can affect
PR metrics independently of model deterioration. Five seeds quantify training
variability, not uncertainty over unseen time periods. This is one historical
dataset with no real-time deployment, production feedback, or analyst review.

## Primary references

- [Original paper and dataset description](https://arxiv.org/abs/1908.02591)
- [PyG EllipticBitcoinDataset implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)
- [Original Kaggle release](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
