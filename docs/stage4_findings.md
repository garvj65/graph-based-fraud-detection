# Stage 4 findings — interview explanation

Validation only; final test and monitoring remain unevaluated.

- “I compared two-layer GCN and GraphSAGE against the same temporal tabular benchmark, keeping unknown nodes in messages and excluding them from loss and metrics.”
- “On validation, GCN with all features reached AP 0.86220 ± 0.01722; GraphSAGE reached 0.94666 ± 0.00658. Neither primary GNN demonstrates a mean validation AP advantage over the strongest tabular result under this compute and tuning budget.”
- “GraphSAGE's message-passing AP lift over its no-message control was +0.07234 ± 0.01312, positive in 5/5 seeds. Messages helped this neural model, while the stronger tabular baseline still performed better. Its supplied neighbor aggregates already contain graph-derived information.”
- “I completed 46 unique fits and reported 40 selected seed results. 2 selected fits reached the epoch cap, so I report the compute limit rather than claiming the architectures were fully optimized.”
- “I reloaded every checkpoint and independently checked AP, alert thresholds and confusion counts. Validation was reused for development, so the future-period comparison and any claim about deterioration after step 43 still need their own runs.”
