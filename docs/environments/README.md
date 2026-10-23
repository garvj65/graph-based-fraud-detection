# Recorded experiment environments

These files preserve the direct dependency versions observed at each experiment
checkpoint. They are historical records, not complete transitive lockfiles or
guarantees of bitwise results across machines. Their contents are unchanged from
the original root-level files.

For normal installation, use the root-level `requirements.txt`,
`requirements-gnn.txt` and `requirements-dev.txt` as described in the
[reproduction guide](../reproduction.md#environment).

| Checkpoint | Observed direct versions |
|---|---|
| Data and split audit | [Stage 1](requirements-stage1-tested.txt) |
| Tabular baselines | [Stage 2](requirements-stage2-tested.txt) |
| Additional graph features | [Stage 3](requirements-stage3-tested.txt) |
| GNN training | [Stage 4](requirements-stage4-tested.txt) |
| Frozen test evaluation | [Stage 5](requirements-stage5-tested.txt) |
| Monitoring and implementation review | [Stage 6](requirements-stage6-tested.txt) |

The recorded experiment used Python 3.12.14 on Linux CPU. To install the final
observed direct versions from the repository root on a compatible machine:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r docs/environments/requirements-stage6-tested.txt
```

The separate checkpoints retain version changes, including NetworkX, rather than
retroactively assigning one environment to every experiment. Stages 4 and 5 have
the same direct package versions but record different experiment checkpoints.
See the [numerical portability review](../final_review.md#exact-score-replay-and-numerical-portability)
before interpreting these files as an exact-replay promise.
