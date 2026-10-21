# Stage 1 findings — interview explanation

These are data and evaluation-design findings from a real audit, not model-performance conclusions.

- “I preserved time as a separate field and enforced training on steps 1–24, validation on 25–34, and testing on 35–49. I also rebuilt PyG's masks because its default training split would include my validation period.”
- “The release has 203,769 transactions and 234,355 directed edges. 77.15% of the labels are unknown, so I keep those nodes in the available graph but exclude them from supervised loss and metrics.”
- “The published 166 features include time. There are 93 local numeric features and 72 supplied neighbor aggregates, so a tree using all numeric inputs already has graph-derived information.”
- “I verified there are no edges across time steps. That means a neighbor-label feature restricted to training labels has no labeled-neighbor coverage in the held-out graphs; I would report that limitation rather than treating hidden test labels as available.”
- “Observed illicit prevalence among known labels is 9.40% in training, 19.77% in validation and 6.50% in test. I will show prevalence and label support alongside later PR metrics, and I have not claimed model decay or a graph-model advantage before training.”
