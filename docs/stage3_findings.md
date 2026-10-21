# Stage 3 findings — interview explanation

All performance numbers are validation-only (steps 25–34); test performance remains unevaluated.

- “I first reran the all-feature XGBoost control on the same five seeds. Its scores, thresholds and metrics matched Stage 2 exactly, so the graph comparison starts from a verified baseline.”
- “Six added structural features gave AP 0.98599 ± 0.00139, compared with 0.98535 ± 0.00180 for the control. The paired AP gain was +0.000647 ± 0.000596, positive in 4 of five seeds. I would describe this as a small development-set lift.”
- “Adding mean/max neighbor scores gave AP 0.98532 ± 0.00137, a paired difference of -0.000028 ± 0.000682 from control. That feature set did not improve validation AP, despite adding complexity.”
- “Structure alone reached AP 0.25526 ± 0.00205. Under the fixed baseline hyperparameters, these six topological inputs were much weaker than the supplied transaction and neighbor features. This is a restricted ablation, not a claim about every graph model.”
- “Raw train-label neighbor features have no coverage in held-out snapshots. I used earlier-step predictions instead, retained unknown nodes, and tested invariance to held-out labels and future edges. The GNN comparison and reliability after step 43 still require their own runs; these validation results do not answer them.”
