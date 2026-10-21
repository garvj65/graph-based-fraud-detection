# Stage 1 correctness review and validation

Reviewed baseline: GitHub commit `d6ba98e806b2ea0f42c6dc2863ff1858c54d3b52`.
Scope: raw loading/ID alignment, labels, temporal masks, induced graph views,
EDA denominators, audit evidence, reproducibility and existing tests.

## Finding fixed

**Low severity — raw timestamp precision could hide invalid fractional times.**
The CSV loader originally narrowed timestamps to float32 before checking that
they were integers. A value such as `24.0000001` can become `24.0`, passing the
check and assigning a malformed row to training. The loader now retains float64
precision for time until validation, while model features remain float32. A
regression test requires this fractional input to fail. This did not change any
time step or split count in the real Elliptic release, whose times are integers.

## Validation performed

- Rehashed all three downloaded raw CSVs and matched the Stage 1 SHA-256 evidence.
- Independently read ID/time columns, joined the class file one-to-one by ID, and
  reproduced the train/validation/test node and class counts.
- Rechecked inclusive 1–24 / 25–34 / 35–49 boundaries, chronological ordering,
  disjointness, unknown exclusion from supervision, retention in available graph
  views, and the absence of cross-time edges.
- Ran the original tests plus the timestamp regression. The expanded Stage 2
  verification additionally checks train-only scaler fitting and class weights,
  future-input invariance, hand-calculated metrics, threshold ties and run logs.

No temporal-mask or label-alignment leakage was found in the reviewed Stage 1
paths. This is a scoped correctness review, not a claim that every future
pipeline path is leakage-free.

## Risks that remain explicit

Supplied forward-neighbor features are retrospective; transaction-arrival
availability cannot be reconstructed from this anonymized release. Unknown
labels are not negatives. No cross-time edges means train-label-only neighbor
features have no labeled-neighbor coverage in validation/test. Full-period EDA
was descriptive; final test performance remains unevaluated during Stage 2.

Stage 2 uses explicit defaults after an unanswered choice form: AP as the primary
PR metric, validation maximum illicit F1 with a higher-threshold tie break, and
94/166 predictors including time. It also runs the 93/165 no-time ablation and
reports trapezoidal PR-AUC separately. No split boundary was changed.
