# Stage 3 — additional graph features

**Validation only: steps 25–34, known labels only. Test remains unevaluated.**

| Model | Inputs | Features | Seeds | AP (PR-AUC) | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | all_control | 166 | 5 | 0.9853 ± 0.0018 | 0.9533 ± 0.0109 | 0.9514 ± 0.0046 | 0.9523 ± 0.0065 |
| XGBoost | all_structure | 172 | 5 | 0.9860 ± 0.0014 | 0.9574 ± 0.0053 | 0.9500 ± 0.0117 | 0.9536 ± 0.0061 |
| XGBoost | all_graph | 174 | 5 | 0.9853 ± 0.0014 | 0.9501 ± 0.0126 | 0.9524 ± 0.0049 | 0.9512 ± 0.0051 |
| XGBoost | structure_only | 6 | 5 | 0.2553 ± 0.0020 | 0.2432 ± 0.0038 | 0.8758 ± 0.0421 | 0.3804 ± 0.0038 |

Every variant uses the Stage 2 selected XGBoost parameters without retuning. Thresholds maximize validation illicit F1 and remain frozen for later test use. Precision/recall/F1 describe validation selection, not a final test estimate.

## Feature definitions

Six structural inputs: directed total degree, in-degree, out-degree, directed PageRank, PageRank multiplied by current snapshot node count, and clustering on the undirected projection. PageRank normalization uses one snapshot, excluding future steps. Unknown nodes stay in these snapshots. No test graph statistics are computed.

Two neighbor-score inputs: mean/max over the unique union of predecessors and successors, excluding the center node. A fixed balanced LogReg on 93 local numeric features generates scores from earlier training steps only; validation uses a frozen steps-1–24 helper. Its scaler fits earlier known labels only. First-step/insufficient-class cases use a configured 0.5 cold-start score, and isolates use a configured zero sentinel. Helpers are shared deterministic feature generators across downstream seeds, not five independent upstream ensembles. Scores have not been calibrated as illicit probabilities.

Raw train-label neighbor coverage is diagnostic only. Held-out snapshots have no edges to train-labeled nodes, so naive neighbor-label features are unavailable there. They are excluded from model inputs. Every helper fit/cutoff is logged separately.

## Paired validation lift

Each row subtracts the same seed's freshly fitted all-feature control. Positive AP means higher validation average precision.

| Variant | Seeds | AP delta mean ± std | F1 delta mean ± std | Positive AP seeds |
|---|---:|---:|---:|---:|
| All + structure + neighbor scores | 5 | -0.0000 ± 0.0007 | -0.0012 ± 0.0016 | 2 |
| All + structure | 5 | +0.0006 ± 0.0006 | +0.0013 ± 0.0030 | 4 |
| Structure only | 5 | -0.7301 ± 0.0034 | -0.5719 ± 0.0057 | 0 |

![Validation comparison and paired AP lift](figures/graph_comparison.png)

## Limits

- Validation has been used for development; final test performance is deferred.
- No GNN was trained in Stage 3.
- Completed-snapshot features are retrospective, not transaction-arrival features.
- Past train labels are assumed available by the next step; label delays are unknown.
- Forward-chained score helpers change training sizes; cold start uses a fixed prior.
- Seed std measures fitting variation, not temporal uncertainty or significance.
- Adding columns changes stochastic feature subsampling; paired lift is descriptive.

The control already includes supplied neighbor aggregates. This comparison measures incremental evidence from the new inputs. Structure-only is a deliberately restricted ablation, not an optimized graph-only model. A GNN comparison remains Stage 4.

## Measured interpretation

Structure adds a small validation AP difference of +0.000647 ± 0.000596 across paired seeds (4/5 positive). The combined structure/neighbor-score variant differs from control by -0.000028 ± 0.000682 (2/5 positive). The simple neighbor scores add no validation gain in this experiment. The GNN and final test comparisons remain pending.
