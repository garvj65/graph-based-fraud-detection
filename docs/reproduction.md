# Reproduce and inspect the benchmark

The repository contains completed experiment logs, tables, representative score
evidence and monitoring reports. Reading those files requires no model execution.
Raw data, weights, scalers and full prediction arrays are regenerated locally.

## Inspect the recorded evidence

| Question | Evidence |
|---|---|
| How do the models compare on future data? | [Test comparison](../results/stage5/test_comparison.csv), [all variants](../results/stage5/test_summary_all_variants.csv), [per-seed metrics](../results/stage5/test_per_seed.csv) |
| What happens around step 43? | [Monitoring report](../results/stage6/REPORT.md), [per-step metrics](../results/stage6/per_step_metrics.csv), [pooled periods](../results/stage6/regime_per_seed.csv) |
| How are alert budgets and unknown labels handled? | [Budget summary](../results/stage5/budget_summary.csv), [evaluation protocol](stage5_protocol.md) |
| What was checked independently? | [Evaluation audit](../results/stage5/independent_validation.json), [monitoring audit](../results/stage6/independent_validation.json), [technical review](final_review.md) |

The representative compressed CSVs retain known-label test scores and all-node
queue evidence for seed 0. When reproducing score/cutoff ties with pandas, use
`read_csv(..., float_precision="round_trip")` and read transaction IDs as strings.

## Environment

Python 3.10+ is supported by CI. The recorded full CPU benchmark used Python 3.12.
From a fresh checkout:

```bash
git clone https://github.com/garvj65/graph-based-fraud-detection.git
cd graph-based-fraud-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-gnn.txt -r requirements-dev.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; on Command
Prompt, use `.venv\Scripts\activate.bat`. Use the Python commands below if Make
is unavailable. The recorded experiment uses CPU; optional CUDA experiments
require a separately configured environment and their own run provenance.

For the recorded direct versions on a compatible Python 3.12/Linux CPU setup,
install the CPU Torch wheel first, then use:

```bash
python -m pip install -r requirements-stage6-tested.txt
```

This records direct observed versions, not a transitive environment lock or a
cross-hardware bitwise guarantee. Requirements with version ranges support a
wider Python/platform matrix; a new environment can produce a new experiment.

## Run the full experiment serially

```bash
make -j1 stage1 stage2 stage3 stage4 stage5 stage6
```

Later targets consume earlier models and manifests. Do not run them concurrently.
A fresh checkout cannot start with `make stage5` or `make stage6`: committed run
records refer to intentionally uncommitted model/score artifacts. Reproducing the
earlier targets writes new manifests and generates those artifacts.

Full five-seed replication performs many model fits. Runtime depends on the host;
no fast-run duration is promised. The feature CSV alone is approximately 690 MB,
with additional space needed for download archives, parsed arrays, model
artifacts and plots. No GPU is required by the default configuration.

Equivalent entry points:

| Work | Command |
|---|---|
| Download, load and audit data | `python -m src.data --config configs/default.yaml --download` |
| Tabular experiment | `python -m src.train --config configs/stage2.yaml --download` |
| Additional graph-feature experiment | `python -m src.graph_train --config configs/stage3.yaml --download` |
| GNN experiment | `python -m src.gnn_train --config configs/stage4.yaml --download` |
| Validate GNN checkpoints | `python scripts/validate_stage4.py` |
| Generate GNN interpretation | `python scripts/summarize_stage4.py` |
| Frozen test evaluation | `python -m src.final_evaluate --config configs/stage5.yaml` |
| Validate test evidence | `python scripts/validate_stage5.py` |
| Generate test interpretation | `python scripts/summarize_stage5.py` |
| Temporal monitoring | `python -m src.monitor --config configs/stage6.yaml` |
| Validate monitoring | `python scripts/validate_stage6.py` |
| Generate monitoring interpretation | `python scripts/summarize_stage6.py` |

The graph experiment requires the reviewed all-feature XGBoost candidate from
the tabular experiment. If a new environment changes that selection, the guard
stops the comparison. Review and log the changed experiment; do not force a
candidate index merely to imitate historical results. The temporal split and
metric policies remain fixed throughout.

For a smaller training example, with explicitly incomplete seed replication:

```bash
python -m src.train --model xgboost --seed 0 --output-dir results/xgboost_seed0
python -m src.train --model gcn --seed 0 --output-dir results/gcn_seed0
```

These commands use the audited full dataset. Their outputs are separate from
the completed benchmark and must not be presented as five-seed results.

## Dataset routes

The exercised full-data route loads raw CSVs acquired from the public PyG mirror.
Kaggle CSVs with the same layout can be supplied directly:

```bash
python -m src.data --raw-dir /path/to/elliptic_bitcoin_dataset
```

The PyG acquisition adapter is implemented but was not exercised for the recorded
full-data run:

```bash
python -m src.data --source pyg
```

Its default processed training mask is not the project's split. This project
recovers time from raw data, joins labels by transaction ID, and enforces train
1–24 / validation 25–34 / test 35–49. The stored matrix has 165 numeric columns;
explicit inclusion of time creates the 166-feature input convention.

## Tests and audits

```bash
python -m pytest -q
python -m ruff check .
```

Tests use small synthetic fixtures, including genuine tiny fits; they do not
download the real dataset or generate benchmark claims. GNN tests require the
GNN stack. The reviewed full-stack implementation passed 127 tests. Hosted CI
covers Python 3.10/3.12 both with and without that stack.

The independent real-data validators require the corresponding artifacts. They
check hashes, training-only preprocessing, original thresholds and score/metric
evidence, rather than accepting summary CSVs on trust. Monitoring reuses saved
test predictions; its training reference builder must reproduce the original
selected model's training input fingerprint.

The original strict evaluation audit passed in its execution environment. A
fresh-runtime check of the original GNN artifacts subsequently failed bitwise
comparison because of small numerical score differences. A separate descriptive
check records those differences without relaxing the original validator:

```bash
python scripts/audit_replay_portability.py
```

The recorded maximum score difference was about 0.0000021. No frozen-threshold
classifications, top-100 memberships or known-illicit queue counts changed.
Original results remain canonical. See the [failure and portability evidence](
../results/final_review) and [review explanation](final_review.md#exact-score-replay-and-numerical-portability).

## Presentation assets

The README title card and reliability figure are generated from committed result
tables, without training or selecting a model:

```bash
python scripts/build_readme_assets.py
```

Outputs live in `docs/assets/`; `readme_evidence.json` records the source-table
hashes and the values displayed in the title card. Historical experiment tables
and logs remain unchanged by that command.
