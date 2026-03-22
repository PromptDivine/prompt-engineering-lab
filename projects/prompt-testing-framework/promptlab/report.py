"""
promptlab/report.py
===================
Results formatting, export, and visualization.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ReportBuilder:
    """
    Builds and exports evaluation reports from BatchRun results.

    Usage:
        report = ReportBuilder(batch_run)
        report.to_csv("results/my_run.csv")
        report.to_json("results/my_run.json")
        report.print_summary()
        report.plot(output_path="results/charts.png")
    """

    def __init__(self, batch_run):
        self.batch_run = batch_run

    def to_csv(self, path: str) -> Path:
        import pandas as pd
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.batch_run.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"CSV saved → {path}")
        return path

    def to_json(self, path: str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.batch_run.run_id,
            "results": [r.to_dict() for r in self.batch_run.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved → {path}")
        return path

    def _clean_df(self):
        """Return a deduplicated, error-free DataFrame."""
        df = self.batch_run.to_dataframe()
        # Drop duplicate columns (latency_s etc.) keeping first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
        df_clean = df[df["error"].isna() | (df["error"] == "")]
        return df_clean

    def print_summary(self):
        """Print a compact leaderboard to console."""
        df_clean = self._clean_df()
        df = self.batch_run.to_dataframe()

        score_cols = [c for c in df_clean.columns
                      if c.startswith(("rubric_", "judge_", "composite"))
                      and df_clean[c].dtype in ["float64", "float32"]]

        print(f"\n{'='*60}")
        print(f"BATCH RUN: {self.batch_run.run_id}")
        print(f"Results: {len(df)} total | {len(df_clean)} successful")
        print(f"{'='*60}")

        group_cols = score_cols + ["latency_s"] if score_cols else ["latency_s"]
        summary = df_clean.groupby(["model", "prompt_id"])[group_cols].mean().round(3)
        print(summary.to_string())
        print()

    def leaderboard(self, metric: str = "composite") -> "pd.DataFrame":
        """Return a leaderboard DataFrame sorted by metric."""
        import pandas as pd
        df_clean = self._clean_df()

        if metric not in df_clean.columns:
            candidates = [c for c in df_clean.columns if metric in c]
            metric = candidates[0] if candidates else "latency_s"

        # Avoid duplicating latency_s when metric is latency_s
        select_cols = ["latency_s"] if metric == "latency_s" else [metric, "latency_s"]
        select_cols = [c for c in select_cols if c in df_clean.columns]

        lb = (
            df_clean
            .groupby(["model", "prompt_id"])[select_cols]
            .mean()
            .round(4)
            .reset_index()
            .sort_values(metric, ascending=False)
        )
        lb.insert(0, "rank", range(1, len(lb) + 1))
        return lb

    def plot(self, output_path: str = "results/charts.png", show: bool = False):
        """Generate a comparison chart from batch results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot")
            return

        df = self.batch_run.to_dataframe()
        df_clean = self._clean_df()

        score_cols = [c for c in df_clean.columns
                      if c.startswith(("rubric_score", "judge_overall", "composite"))
                      and df_clean[c].dtype in ["float64", "float32"]]

        PALETTE = {
            "bg": "#0f1117", "surface": "#161820", "border": "#1e2130",
            "text": "#f0f2f8", "muted": "#5a6080", "accent": "#e8ff47",
        }
        MODEL_COLORS = [
            "#47c8ff", "#e8ff47", "#ff8c47", "#b847ff", "#47ffb2", "#ff4776"
        ]

        plt.rcParams.update({
            "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["border"], "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"], "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"], "text.color": PALETTE["text"],
            "grid.color": PALETTE["border"], "font.family": "monospace",
        })

        n_panels = 2 if score_cols else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        if n_panels == 1:
            axes = [axes]

        models = df_clean["model"].unique()
        prompt_ids = df_clean["prompt_id"].unique()
        x = np.arange(len(prompt_ids))
        width = 0.7 / max(len(models), 1)

        # Panel 1: Score comparison
        if score_cols:
            primary_metric = score_cols[0]
            ax = axes[0]
            pivot = df_clean.groupby(["prompt_id", "model"])[primary_metric].mean().unstack(fill_value=0)
            pivot = pivot.reindex(prompt_ids)
            for i, model in enumerate(pivot.columns):
                offset = (i - len(pivot.columns)/2) * width + width/2
                ax.bar(x + offset, pivot[model], width=width*0.9,
                       color=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.85, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels(prompt_ids, rotation=15, ha="right")
            ax.set_ylabel(primary_metric)
            ax.set_title(f"SCORE: {primary_metric}", loc="left", fontsize=10)
            ax.legend(framealpha=0, fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Panel 2: Latency
        ax = axes[-1]
        lat_pivot = df_clean.groupby(["prompt_id", "model"])["latency_s"].mean().unstack(fill_value=0)
        lat_pivot = lat_pivot.reindex(prompt_ids)
        for i, model in enumerate(lat_pivot.columns):
            offset = (i - len(lat_pivot.columns)/2) * width + width/2
            ax.bar(x + offset, lat_pivot[model], width=width*0.9,
                   color=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.85, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_ids, rotation=15, ha="right")
        ax.set_ylabel("Latency (s)")
        ax.set_title("LATENCY (s)", loc="left", fontsize=10)
        ax.legend(framealpha=0, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle(f"promptlab — Run: {self.batch_run.run_id}",
                     fontsize=12, fontweight="bold", color=PALETTE["text"])

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["bg"], dpi=150)
        if show:
            plt.show()
        plt.close()
        logger.info(f"Chart saved → {out_path}")
        return out_path
