# Temporal monitoring protocol

Stage 6 reuses Stage 5's frozen predictions and the candidate chosen by mean
validation AP. It does not alter the 1–24 / 25–34 / 35–49 split, model weights,
scalers, features, decision thresholds or seed selection. The README/About
redesign remains deferred until review.

The policy choice form returned no selections. Before running this analysis,
the explicit default was recorded: all-node PSI/KS drift plus a supported,
three-step pooled-F1 warning. Test was already opened in Stage 5, so even a
reference frozen before this analysis does not make the warning rules
prospectively validated. No monitoring threshold is optimized on test results.

## Populations and metrics

- AP means sklearn average precision. Classification metrics and confusion
  counts exclude unknown labels. Undefined AP/recall/F1 with no illicit support
  remain null; no known labels makes all quality rates null.
- Alert rate and mean score use all transactions. Label coverage is known/all;
  illicit rate is illicit/known, with known-illicit/all also retained. Unknown
  alerts never become false positives.
- Per-step results cover all seven primary variants and all five seeds, using
  the exact Stage 5 score arrays. Mean ± sample std measures seed variation.
- Training references use all training nodes, including unknowns, to match
  the all-node monitoring population. Models were fitted on known labels only.
  Their training scores are in-sample, so drift can reflect fitting optimism
  as well as time shift. Validation drift is shown as a sanity check.

## Drift definitions

Requested PSI bins: ten training quantile bins, repeated cuts collapsed, open
outer bins. A constant variable uses three bins around its training value so
changes in either direction remain observable. Each probability is clipped to
1e-6 and renormalized before summing `(current-reference) * log(current/reference)`.
The cuts and reference counts never adapt to validation/test. Sample sizes,
actual bin counts, training ranges and out-of-training-range rates are retained.

KS uses the two-sided empirical-CDF maximum distance, with SciPy's asymptotic
p-value recorded only as a diagnostic. The graph creates dependent samples;
scores/features are tied/discrete and comparisons are repeated. P-values do not
trigger action or provide causal evidence.

Drift is computed for the validation-selected candidate, across five seeds for
scores. The top ten features are chosen once from the predeclared seed-0 model's
training-tree gain, with feature-name ties. Features are anonymized and gain is
not causal importance. If time is selected, its deterministic advance must be
distinguished from unexpected input failure.

## Warning policy and retraining review

PSI > 0.2 is a heuristic investigation warning. Two consecutive test steps above
it indicate persistent shift. The supervised warning pools the latest three
test steps using the original score cutoff, requires a full window with at
least 100 known labels and 20 illicit labels, and checks whether pooled F1 is
below 80% of that seed's saved validation F1. Pooling computes F1 from combined
confusion counts, not the mean of the three F1 values. These support thresholds
and the fraction are declared heuristics, not tuned operational guarantees.

Drift alone prompts pipeline checks and targeted adjudication. A supported F1
warning prompts a retraining review. Recent labels, actual label delays and a
new chronological challenger evaluation are required before any promotion.
This code never retrains, changes cutoffs or automatically promotes a model.
Supervised warnings are retrospective because label-availability times are
absent. A step-level warning therefore has no proven live detection latency.

## Reported event

The original paper identifies a dark-market shutdown at step 43. Fixed periods
35–42 and 43–49 are pooled separately as descriptive cohorts. This event boundary
comes from the project brief and literature, not a fitted test change point.
Performance, observed prevalence, label support and drift are inspected together.
There is no causal counterfactual, and no claim that retraining fixes the event.

Source: Weber et al., 2019, section 4, “The Dark Market Shutdown”,
<https://arxiv.org/pdf/1908.02591>. KS reference:
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>.

## Reproduction and audit

With earlier model/score artifacts available, run `make stage6`. A fresh checkout
must reproduce prior stages first. The stage refuses changed/missing artifacts;
it never silently refits. Training structure is reconstructed per snapshot and
its known-label feature fingerprint must match the original fit exactly. No
prefix-24 score helper is applied to training rows. A hashed reference containing
train bins, training feature ranking and validation F1 floors is saved before
test monitoring output is computed.

The independent audit reconstructs histograms and PSI/KS distances, reloads the
five candidate models for training score references, verifies training feature
fingerprints, reconstructs test metrics and rolling counts, checks warning rules
and aggregate summaries, and confirms that source evidence and thresholds are
unchanged. The prior Stage 5 audit separately reloads all 100 frozen models.
