# Stage 2 findings — interview explanation

Validation only: these do not describe test or production performance.

- “I trained only on known labels from steps 1–24 and used steps 25–34 for model and threshold selection. I kept the test period unevaluated so later graph experiments still have an honest final comparison.”
- “XGBoost with all 166 inputs achieved validation AP 0.9853 ± 0.0018 and illicit F1 0.9523 ± 0.0065 across five seeds. Those are development scores because validation also selected the candidate and threshold.”
- “Local-only XGBoost achieved AP 0.9829 ± 0.0005. Adding the supplied neighbor aggregates changed mean AP by +0.0025; this is measured incremental evidence, not a claim that a GNN wins.”
- “I checked whether time was driving the result: the 165-input no-time XGBoost ablation reached AP 0.9857 ± 0.0011. I kept the original main protocol rather than silently choosing whichever convention scored higher.”
- “I logged every fit and froze per-seed thresholds using validation only. I independently recomputed scores and thresholds and reloaded the saved models. Small seed variance does not show temporal reliability; drift and post-step-43 performance remain for the monitoring stage.”
