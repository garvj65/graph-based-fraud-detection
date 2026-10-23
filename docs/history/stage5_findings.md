# Stage 5 interview findings

- “I froze every model, preprocessing step, alert policy and threshold before evaluating steps 35–49. Unknown nodes stayed in graph inference and stayed out of classification metrics.”
- “The all-feature XGBoost baseline reached test AP 0.7805 ± 0.0026, compared with GraphSAGE 0.4235 ± 0.0460 and GCN 0.3421 ± 0.0584. These GNNs did not outperform the strong tabular model.”
- “Structural features added paired AP +0.003255 ± 0.002646, but did not improve mean F1. I would describe this as a small ranking gain, rather than claim additional graph complexity delivered a clear business benefit.”
- “A 100-alert-per-step all-transaction queue identified 390 known illicit transactions on average over 1500 reviews. About 73.2% of alerts were unlabeled, so I cannot claim overall operational precision.”
- “The selected model moved from validation AP 0.9860 to test AP 0.7837. The next task is to locate that deterioration with per-step metrics and drift checks, rather than assume a particular event caused it.”
