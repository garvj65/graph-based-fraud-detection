# Five findings to explain in an interview

- I built a chronological fraud benchmark with train 1–24, validation 25–34 and test 35–49. Unknown transactions participate in the graph but never become negative labels or enter supervised loss/quality metrics.
- A strong all-feature XGBoost baseline achieved test AP 0.7805 ± 0.0026; GraphSAGE reached 0.4235 ± 0.0460. The tested GNNs did not beat the tabular baseline over the whole future period.
- Added structural features reached AP 0.7837 ± 0.0035, without improving mean F1. That small ranking gain does not establish that extra graph complexity pays for itself.
- At 100 reviews per step, the selected model's all-node queue found 390 ± 14 known illicit transactions over 1500 reviews. 73.2% of alerts were unlabeled, so overall operational precision remains unknown.
- Monitoring exposed F1 dropping from 0.857 in steps 35–42 to 0.049 in steps 43–49. PSI alone was not selective; I proposed supported performance warnings, label investigation and a new temporal challenger evaluation before any retraining promotion.
