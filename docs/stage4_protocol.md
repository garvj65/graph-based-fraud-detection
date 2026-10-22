# Stage 4 protocol

This stage preserves the reviewed split and metric definitions. Training uses
steps 1–24, validation uses 25–34, and test model scoring and graph features are
deferred to Stage 5. The protocol is configured in `configs/stage4.yaml`.

## Models and inputs

- GCN: two `GCNConv` layers, symmetric normalization and self-loops, no cached
  adjacency. Caching a training adjacency and reusing it for validation would
  be incorrect because those graphs have different node memberships.
- GraphSAGE: two mean-aggregation `SAGEConv` layers, with root transformations.
- Both use ReLU and dropout between layers and one output logit per node.
- Local: 94 inputs including time, preserving Stage 2's primary convention.
- All: 166 inputs including time and the supplied 72 neighbor aggregates.
- Structure-only: Stage 3's six structural features, computed within each
  completed snapshot. No transaction attributes, labels or time input.

Messages use the unique bidirectional union of payment edges within each
snapshot. This is an explicit modeling choice: messages do not distinguish
payment direction. The structure-only inputs still contain directed in/out
degree and PageRank. No future node or edge reaches the training graph.

All training snapshots form one disconnected full-batch graph; validation
snapshots form a different disconnected graph. These are retrospective
completed-snapshot experiments, not an arrival-time replay. Unknown nodes
participate in messages but never become negative examples.

## Fitting and selection

The StandardScaler fits the 23,606 known training rows, matching the tabular
preprocessing scope. It transforms all available training/validation nodes.
Weighted BCE uses a training-only positive weight of `21,387 / 2,219`.
Unknown targets are removed before loss evaluation. Labels are never model
inputs or neighbor aggregates.

The two predeclared candidates each use 64 hidden channels: learning rate
0.01/dropout 0.3, or learning rate 0.003/dropout 0.5. Adam weight decay is
0.0005, gradient norm is clipped at 5, the maximum is 250 epochs, and patience
is 30 epochs. Each architecture/input variant chooses a candidate on seed 0
validation AP, then freezes it for seeds 0–4. The winning seed-0 fit is reused.
This is a small compute budget, not an exhaustive architecture search.

Checkpoint selection maximizes validation average precision, with exact ties
keeping the earliest epoch. Validation runs without gradients or dropout.
After restoring the selected checkpoint, the maximum-F1 validation threshold
is selected once, with ties going to the higher cutoff. Each run's cutoff is
frozen for later test use. There is no train+validation refit or best-seed choice.

For each architecture, an additional all-feature no-message control removes
inter-node edges and reuses the selected all-feature candidate without tuning
another candidate. It independently selects its checkpoint and threshold on
validation. GCN retains self-loops; GraphSAGE retains its root transformation.
This provides a narrower check of the implemented message contribution than
comparing a neural model with a tree alone.

The complete default run requires 46 unique fits: 12 candidate trials, 24
additional seed replications, and 10 no-message controls. It reports 40 selected
seed results: two architectures × four input/message variants × five seeds.
Sample standard deviation uses ddof=1 and measures training randomness on one
temporal split, not uncertainty about future fraud.

## Reproduction and evidence

Install the CPU stack, then run the complete experiment and artifact validation:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-gnn.txt
make stage4
```

The requested individual-model interface is also available:

```bash
python -m src.train --model gcn --seed 0 --output-dir results/stage4_gcn_seed0
python -m src.gnn_train --model graphsage --device cuda --output-dir results/stage4_cuda
```

The documented default experiment uses CPU. A CUDA run is an optional separate
experiment, not an assertion of measured GPU performance. Deterministic
algorithms are requested; unavailable deterministic operations fail explicitly.
Floating-point identity across different machines or dependency versions is
not guaranteed. Single-seed summaries report unavailable std.

If an execution session is interrupted, the batch can be resumed:

```bash
python scripts/resume_stage4.py --batch results/stage4/batches/<batch_id>.json --workers 2
python scripts/validate_stage4.py
python scripts/summarize_stage4.py
```

The recorded experiment resumed after 32 completed fits. Those artifacts were
hash-checked and reused. One unfinished attempt was archived, excluded from
selection/metrics, and restarted with its original seed. The remaining fits
used two independent CPU workers, each retaining the configured four threads.
No data, hyperparameter, checkpoint criterion or threshold policy changed.
The batch logs the resumption script hash and worker settings. A clean run
still requires 46 completed fits; interruption attempts have no final metrics.

Run JSONs log effective configs, seeds, class weights, feature/graph/scaler
fingerprints, source/data versions and hashes, best epochs, metrics and artifact
hashes. Epoch histories and representative validation scores are committed.
Model weights, scalers and full score arrays are stored under ignored
`artifacts/` and regenerated by reproduction. No raw dataset is committed.

`scripts/validate_stage4.py` reloads every checkpoint, compares saved scores,
checks input/source hashes and train-only scaling, and independently recomputes
AP, thresholds and confusion counts with NumPy. It also verifies candidate and
epoch selection, seed summaries, and message/no-message deltas. It never scores
test data. The test suite uses small synthetic fixtures, not result-producing
subsets of the benchmark. CI includes CPU GNN checks on Python 3.10 and 3.12.

## Interpretation boundaries

Validation has been reused for settings, epoch selection and thresholds. It is
an optimistic development estimate. The all-feature baseline already includes
graph-derived information. GraphSAGE uses full neighborhoods here, not neighbor
sampling. Two layers, one edge-direction policy and a limited tuning budget
cannot establish a universal result about GNNs. Runs still improving at the
epoch cap must be identified in the results report.

Unknown-label selection bias, absent label timestamps, anonymized supplied
features and completed-snapshot availability remain limitations. Uncalibrated
scores are not illicit probabilities. A final future-period comparison and
monitoring claims await Stages 5 and 6.

## Implementation references

- [PyG 2.6.1 GCNConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GCNConv.html)
- [PyG 2.6.1 SAGEConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.SAGEConv.html)
- [PyTorch 2.8 BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
