# Stage 2 — tabular baseline results

**Validation only: steps 25–34. Test performance is not evaluated.**

Mean ± sample std across the listed seeds; no accuracy headline. Each seed's precision/recall/F1 uses its own validation-selected maximum-F1 threshold. Those thresholds are recorded and frozen for later test use.

## Primary feature convention

| Model | Inputs | Features | Seeds | AP (PR-AUC) | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LogReg | all | 166 | 5 | 0.6115 ± 0.0000 | 0.6327 ± 0.0000 | 0.8053 ± 0.0000 | 0.7087 ± 0.0000 |
| LogReg | local | 94 | 5 | 0.5944 ± 0.0000 | 0.6730 ± 0.0000 | 0.7353 ± 0.0000 | 0.7028 ± 0.0000 |
| XGBoost | all | 166 | 5 | 0.9853 ± 0.0018 | 0.9533 ± 0.0109 | 0.9514 ± 0.0046 | 0.9523 ± 0.0065 |
| XGBoost | local | 94 | 5 | 0.9829 ± 0.0005 | 0.9611 ± 0.0169 | 0.9176 ± 0.0176 | 0.9386 ± 0.0013 |

## Time-feature ablation

| Model | Inputs | Features | Seeds | AP (PR-AUC) | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LogReg | all | 165 | 5 | 0.6115 ± 0.0000 | 0.6327 ± 0.0000 | 0.8037 ± 0.0000 | 0.7080 ± 0.0000 |
| LogReg | local | 93 | 5 | 0.6980 ± 0.0000 | 0.7383 ± 0.0000 | 0.8126 ± 0.0000 | 0.7736 ± 0.0000 |
| XGBoost | all | 165 | 5 | 0.9857 ± 0.0011 | 0.9550 ± 0.0101 | 0.9538 ± 0.0073 | 0.9544 ± 0.0033 |
| XGBoost | local | 93 | 5 | 0.9828 ± 0.0006 | 0.9685 ± 0.0092 | 0.9075 ± 0.0077 | 0.9370 ± 0.0007 |

## Protocol and interpretation

Predeclared candidate counts per feature variant: {'logreg': 3, 'xgboost': 3}. Candidates are compared on fixed tuning seed 0 using validation PR ranking performance. Winners are frozen and replicated on seeds [0, 1, 2, 3, 4]; the tuning-seed fit is reused when requested. No best-seed selection, test-based tuning, random split, SMOTE, or train+validation refit occurs. Exact candidate counts/seeds come from the saved configuration; smaller CLI runs may intentionally use fewer candidates or seeds.

Average precision and trapezoidal PR-AUC are separate columns in the CSV files. Class weights and XGBoost scale_pos_weight use known training labels only. LogReg's StandardScaler fits known training rows only; XGBoost's histogram bins are constructed without an evaluation set. Weighted outputs are uncalibrated scores.

LogReg uses deterministic LBFGS, so repeating its seed need not produce variation. XGBoost's row/column subsampling provides seed-dependent variation. A small std does not establish robustness to temporal shift. Validation has been used both for model selection and threshold optimization, so these are selection-biased development estimates. Final test comparison remains Stage 5.

The all-feature baseline includes the supplied neighbor aggregates; it already has graph-derived information. This stage cannot establish whether additional graph features or GNNs improve it.

![Validation PR and threshold diagnostics](figures/validation_diagnostics.png)

The diagnostic uses the configured representative seed 0, never the best-performing seed. Labeled-only alert rates are not all-node analyst workloads. Alert-budget evaluation and operational denominators remain Stage 5 decisions.
