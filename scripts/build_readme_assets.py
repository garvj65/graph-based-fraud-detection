"""Build README visuals from committed experiment tables without fitting any model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "s3_xgboost_all_structure"
ASSETS = ROOT / "docs/assets"


def build_assets() -> None:
    """Export a vector title card, measured reliability plot and source-hash ledger."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    comparison_path = ROOT / "results/stage5/test_comparison.csv"
    period_path = ROOT / "results/stage6/regime_summary.csv"
    step_path = ROOT / "results/stage6/per_step_summary.csv"
    comparison = pd.read_csv(comparison_path).set_index("variant").loc[CANDIDATE]
    periods = pd.read_csv(period_path)
    after = periods[(periods.variant == CANDIDATE) & (periods.period == "event_and_after")]
    missed = 1 - float(after.iloc[0].recall_mean)
    hero = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="390"
  viewBox="0 0 1200 390" role="img" aria-labelledby="title desc">
<title id="title">Graph fraud detection: strong scores, fragile detection</title>
<desc id="desc">The selected model reaches test AP {comparison.average_precision_mean:.4f},
but misses {missed:.1%} of known illicit transactions in steps 43 to 49.
Five training seeds, with a fixed temporal split. The graph motif is decorative.</desc>
<defs>
  <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
    <stop stop-color="#0e1e2a"/><stop offset="1" stop-color="#132f38"/>
  </linearGradient>
  <radialGradient id="glow"><stop stop-color="#48ddbd" stop-opacity=".16"/>
    <stop offset="1" stop-color="#48ddbd" stop-opacity="0"/></radialGradient>
</defs>
<rect width="1200" height="390" rx="18" fill="url(#bg)"/>
<circle cx="1010" cy="135" r="220" fill="url(#glow)"/>
<g fill="none" stroke="#61d4c0" stroke-opacity=".22" stroke-width="1.4">
  <path d="M840 72L929 43L1000 97L1092 62L1148 138L1069 205L966 174L879 211L840 72"/>
  <path d="M840 72L1000 97L966 174L929 43M1000 97L1069 205L1092 62M879 211L1000 97L1148 138"/>
</g>
<g fill="#8ce5d2">
  <circle cx="840" cy="72" r="4"/><circle cx="929" cy="43" r="5"/>
  <circle cx="1000" cy="97" r="7"/><circle cx="1092" cy="62" r="4"/>
  <circle cx="1148" cy="138" r="5"/><circle cx="1069" cy="205" r="4"/>
  <circle cx="966" cy="174" r="5"/><circle cx="879" cy="211" r="4"/>
</g>
<g font-family="DejaVu Sans,Arial,sans-serif">
  <text x="48" y="48" fill="#8da9b7" font-size="14" letter-spacing="3">
    ELLIPTIC / GRAPH FRAUD BENCHMARK</text>
  <text x="46" y="122" fill="#f0f7fa" font-size="61" font-weight="700">
    Strong scores.</text>
  <text x="46" y="190" fill="#7be4cb" font-size="61" font-weight="700">
    Fragile detection.</text>
  <text x="49" y="229" fill="#b4c9d2" font-size="20">
    Tabular models. Graph learning. Temporal reliability.</text>
  <path d="M48 266H1152" stroke="#3b5965" stroke-opacity=".7"/>
  <text x="49" y="321" fill="#f0f7fa" font-size="38" font-weight="700">
    {comparison.average_precision_mean:.3f}</text>
  <text x="50" y="351" fill="#94b0bd" font-size="12" letter-spacing="1.1">
    WHOLE-PERIOD TEST AP</text>
  <text x="415" y="321" fill="#f6b879" font-size="38" font-weight="700">
    {missed:.1%}</text>
  <text x="416" y="351" fill="#94b0bd" font-size="12" letter-spacing=".7">
    KNOWN ILLICIT MISSED / STEPS 43–49</text>
  <text x="872" y="321" fill="#f0f7fa" font-size="38" font-weight="700">
    {int(comparison.n_seeds)} seeds</text>
  <text x="873" y="351" fill="#94b0bd" font-size="12" letter-spacing="1.1">
    FIXED TEMPORAL SPLIT</text>
</g>
</svg>
"""
    (ASSETS / "hero.svg").write_text(hero)
    step = pd.read_csv(step_path)
    step = step[step.variant == CANDIDATE].sort_values("time_step")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained")
    fig.patch.set_facecolor("#f5f8fa")
    for ax, metric, title, upper in zip(
        axes,
        ("f1", "alert_rate_all"),
        ("Detection quality collapses", "The investigation queue continues"),
        (1.0, 0.20),
    ):
        x, mean, std = (
            step[column].to_numpy(float)
            for column in ("time_step", f"{metric}_mean", f"{metric}_std")
        )
        ax.set_facecolor("#f5f8fa")
        ax.plot(x, mean, color="#087e82", linewidth=2.4, marker="o", markersize=3.5)
        ax.fill_between(
            x,
            np.maximum(0, mean - std),
            np.minimum(1, mean + std),
            color="#087e82",
            alpha=0.13,
            linewidth=0,
        )
        ax.axvspan(43, 49, color="#edb26d", alpha=0.13)
        ax.axvline(43, color="#a95b2a", linestyle=(0, (3, 3)), linewidth=1.2)
        ax.set(xlim=(35, 49), ylim=(0, upper), xlabel="Time step", title=title)
        ax.set_xticks([35, 37, 39, 41, 43, 45, 47, 49])
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["left", "bottom"]].set_color("#b2c2ca")
        ax.tick_params(colors="#344f5b", length=3)
        if metric == "f1":
            ax.set_ylabel("Illicit F1 · known labels")
        else:
            ax.set_ylabel("Alerts / all transactions")
            ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle(
        "Frozen XGBoost + structure · mean ± sample std over five training seeds",
        fontsize=12,
        fontweight="bold",
        color="#173846",
    )
    fig.savefig(ASSETS / "reliability.png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(ASSETS / "reliability.svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    ledger = {
        "candidate": CANDIDATE,
        "test_ap_mean": float(comparison.average_precision_mean),
        "post_event_known_illicit_missed_fraction": missed,
        "seeds": int(comparison.n_seeds),
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (comparison_path, period_path, step_path)
        },
        "models_fitted": False,
        "metrics_or_thresholds_changed": False,
    }
    (ASSETS / "readme_evidence.json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    build_assets()
