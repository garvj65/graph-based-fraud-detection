# Monitoring interview findings

- The frozen candidate's F1 changes from 0.8572 before step 43 to 0.0492 afterward; an aggregate test score concealed a severe reliability failure.
- Post-event recall is 4.6%, while the all-node alert rate remains 9.3%; the review queue keeps consuming effort while missing known illicit activity.
- PSI > 0.2 warns during 100% of validation step/seed comparisons, so generic drift thresholds need calibration and cannot substitute for label-based quality monitoring.
- The pooled F1 policy first warns at step 44 for all 5 seeds; this is retrospective because true label delays are unavailable.
- Retraining should be reviewed with recent adjudicated labels, followed by a new temporal challenger evaluation. Neither drift nor this historical event guarantees a successful retrain.
