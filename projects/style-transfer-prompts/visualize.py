"""
visualize.py
============
Style Transfer Prompts — Chart Generator
Project: P2 · prompt-engineering-lab

Charts:
  1. Formality Shift Heatmap     (style × model)
  2. FK Grade Level Comparison   (source vs each style, per model)
  3. Strategy Comparison         (A vs B vs C per style)
  4. Metric Delta Radar          (how much each style shifts each metric)
  5. Compression by Style        (how much each style expands/contracts)
  6. Sentiment Shift             (source → target per style)
  7. Master 4-panel hero chart   → results/charts.png
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":      "#0f1117",
    "surface": "#161820",
    "border":  "#1e2130",
    "text":    "#f0f2f8",
    "muted":   "#5a6080",
    "accent":  "#e8ff47",
    "blue":    "#47c8ff",
    "purple":  "#b847ff",
    "orange":  "#ff8c47",
    "green":   "#47ffb2",
    "red":     "#ff4776",
    "pink":    "#ff47c8",
}

STYLE_COLORS = {
    "journalism":  PALETTE["blue"],
    "academic":    PALETTE["purple"],
    "legal":       PALETTE["orange"],
    "executive":   PALETTE["accent"],
    "casual":      PALETTE["green"],
    "storytelling":PALETTE["pink"],
    "technical":   PALETTE["red"],
    "marketing":   "#ffd147",
    "medical":     "#47ffd4",
    "minimalist":  "#a0a8c8",
}

MODEL_COLORS = {
    "GPT-4o-mini":     PALETTE["blue"],
    "GPT-4o":          PALETTE["accent"],
    "Claude Haiku":    PALETTE["orange"],
    "Claude Sonnet 4.6": PALETTE["purple"],
    "Mistral 7B":      PALETTE["green"],
    "Llama 3 8B":      PALETTE["red"],
}

STYLE_ORDER = ["journalism","academic","legal","executive","casual",
               "storytelling","technical","marketing","medical","minimalist"]

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],
        "axes.labelcolor":  PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],
        "xtick.color":      PALETTE["muted"],
        "ytick.color":      PALETTE["muted"],
        "text.color":       PALETTE["text"],
        "grid.color":       PALETTE["border"],
        "grid.linewidth":   0.5,
        "font.family":      "monospace",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.titleweight": "bold",
        "axes.titlepad":    10,
        "figure.dpi":       150,
    })


def load_data(path: Path):
    df = pd.read_csv(path)
    df_clean = df[df.get("error", pd.Series(dtype=str)).isna() | (df.get("error", pd.Series(dtype=str)) == "")].copy()
    return df_clean


# ── Chart 1: Formality Shift Heatmap ────────────────────────

def chart_formality_heatmap(df: pd.DataFrame, out: Path):
    styles = [s for s in STYLE_ORDER if s in df["style"].values]
    models = df["model"].unique()

    pivot = df.groupby(["style","model"])["formality_score"].mean().unstack(fill_value=0)
    pivot = pivot.reindex([s for s in STYLE_ORDER if s in pivot.index])

    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.5), len(styles)*0.7 + 1.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "black" if val > 0.6 else PALETTE["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Formality Score (0=informal, 1=formal)")
    ax.set_title("FORMALITY SCORE — Style × Model", loc="left")
    plt.tight_layout()

    path = out / "chart_formality_heatmap.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 2: FK Grade Level per Style ───────────────────────

def chart_fk_grade(df: pd.DataFrame, out: Path):
    styles = [s for s in STYLE_ORDER if s in df["style"].values]
    model_avg = df.groupby(["style","model"])["fk_grade"].mean().unstack(fill_value=0)
    model_avg = model_avg.reindex([s for s in STYLE_ORDER if s in model_avg.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_avg.index))
    n = len(model_avg.columns)
    width = 0.75 / n

    for i, model in enumerate(model_avg.columns):
        offset = (i - n/2) * width + width/2
        color = MODEL_COLORS.get(model, PALETTE["blue"])
        ax.bar(x + offset, model_avg[model], width=width*0.9, color=color, alpha=0.85, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(model_avg.index, rotation=15, ha="right")
    ax.set_ylabel("Flesch-Kincaid Grade Level")
    ax.set_title("READABILITY (FK GRADE) — By Style & Model", loc="left")
    ax.axhline(8, color=PALETTE["muted"], linestyle="--", linewidth=0.7, alpha=0.5, label="Grade 8 (general public)")
    ax.axhline(14, color=PALETTE["muted"], linestyle=":", linewidth=0.7, alpha=0.5, label="Grade 14 (academic)")
    ax.legend(framealpha=0, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out / "chart_fk_grade.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 3: Strategy A vs B vs C ───────────────────────────

def chart_strategy_comparison(df: pd.DataFrame, out: Path):
    if "strategy" not in df.columns:
        return

    strat_scores = df.groupby(["strategy", "style"])["formality_score"].mean().unstack(fill_value=0)
    strat_scores = strat_scores.reindex(columns=[s for s in STYLE_ORDER if s in strat_scores.columns])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(strat_scores.columns))
    strategies = strat_scores.index.tolist()
    colors_strat = [PALETTE["accent"], PALETTE["blue"], PALETTE["purple"]]
    width = 0.75 / len(strategies)

    for i, strat in enumerate(strategies):
        offset = (i - len(strategies)/2) * width + width/2
        ax.bar(x + offset, strat_scores.loc[strat], width=width*0.9,
               color=colors_strat[i % len(colors_strat)], alpha=0.85, label=strat)

    ax.set_xticks(x)
    ax.set_xticklabels(strat_scores.columns, rotation=15, ha="right")
    ax.set_ylabel("Avg Formality Score")
    ax.set_title("PROMPT STRATEGY COMPARISON (A=direct / B=role / C=contrastive)", loc="left")
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out / "chart_strategy_comparison.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 4: Metric Delta Radar per Style ───────────────────

def chart_delta_radar(df: pd.DataFrame, out: Path):
    delta_cols = ["delta_fk_grade", "delta_formality", "delta_sentiment",
                  "delta_sentence_length", "delta_ttr"]
    available = [c for c in delta_cols if c in df.columns]
    if len(available) < 3:
        return

    style_deltas = df.groupby("style")[available].mean()
    style_deltas = style_deltas.reindex([s for s in STYLE_ORDER if s in style_deltas.index])

    # Normalize each delta to -1..1 for radar
    for col in available:
        max_abs = style_deltas[col].abs().max()
        if max_abs > 0:
            style_deltas[col] = style_deltas[col] / max_abs

    N = len(available)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    labels = [c.replace("delta_", "").replace("_", " ").upper() for c in available]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=8)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels(["-0.5", "0", "+0.5"], color=PALETTE["muted"], fontsize=7)
    ax.grid(color=PALETTE["border"], linewidth=0.8)
    ax.spines["polar"].set_color(PALETTE["border"])
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.5, alpha=0.5)

    for style, row in style_deltas.iterrows():
        vals = row[available].tolist()
        vals += vals[:1]
        color = STYLE_COLORS.get(style, PALETTE["blue"])
        ax.plot(angles, vals, "o-", color=color, linewidth=1.8, markersize=4, label=style)
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_title("METRIC SHIFTS — How Each Style Changes the Text\n(normalized, 0 = no change vs source)",
                 y=1.12, fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), framealpha=0, fontsize=8)

    path = out / "chart_delta_radar.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 5: Compression by Style ───────────────────────────

def chart_compression(df: pd.DataFrame, out: Path):
    comp = df.groupby("style")["compression_ratio"].mean().reindex(
        [s for s in STYLE_ORDER if s in df["style"].values]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [STYLE_COLORS.get(s, PALETTE["blue"]) for s in comp.index]
    bars = ax.bar(comp.index, comp.values, color=colors, alpha=0.85, width=0.6)

    ax.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=1.2,
               alpha=0.7, label="Source length (1.0)")
    for bar, val in zip(bars, comp.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=8,
                color=PALETTE["text"])

    ax.set_ylabel("Compression Ratio (vs source)")
    ax.set_title("OUTPUT LENGTH vs SOURCE — By Style\n(<1.0 = shorter, >1.0 = longer)", loc="left")
    ax.legend(framealpha=0, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(comp.index, rotation=20, ha="right")

    path = out / "chart_compression.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Master hero chart ────────────────────────────────────────

def chart_master(df: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    styles_present = [s for s in STYLE_ORDER if s in df["style"].values]

    # Panel 1: Formality by style
    form = df.groupby("style")["formality_score"].mean().reindex(styles_present)
    colors1 = [STYLE_COLORS.get(s, PALETTE["blue"]) for s in form.index]
    ax1.barh(form.index, form.values, color=colors1, height=0.6, alpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("FORMALITY BY STYLE", loc="left", fontsize=9)
    ax1.set_xlabel("Formality Score")
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: FK Grade by style
    fk = df.groupby("style")["fk_grade"].mean().reindex(styles_present)
    colors2 = [STYLE_COLORS.get(s, PALETTE["blue"]) for s in fk.index]
    ax2.barh(fk.index, fk.values, color=colors2, height=0.6, alpha=0.9)
    ax2.axvline(8, color=PALETTE["muted"], linestyle="--", linewidth=0.8, alpha=0.6, label="Grade 8")
    ax2.axvline(14, color=PALETTE["accent"], linestyle=":", linewidth=0.8, alpha=0.6, label="Grade 14")
    ax2.set_title("READING LEVEL (FK GRADE)", loc="left", fontsize=9)
    ax2.set_xlabel("Grade Level")
    ax2.legend(framealpha=0, fontsize=7)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: Compression
    comp = df.groupby("style")["compression_ratio"].mean().reindex(styles_present)
    colors3 = [STYLE_COLORS.get(s, PALETTE["blue"]) for s in comp.index]
    ax3.bar(comp.index, comp.values, color=colors3, alpha=0.85, width=0.6)
    ax3.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_title("OUTPUT LENGTH vs SOURCE", loc="left", fontsize=9)
    ax3.set_ylabel("Compression Ratio")
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticklabels(comp.index, rotation=30, ha="right", fontsize=7)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Strategy comparison (formality A/B/C)
    if "strategy" in df.columns:
        strat = df.groupby(["strategy","style"])["formality_score"].mean().unstack(fill_value=0)
        strat = strat.reindex(columns=[s for s in styles_present if s in strat.columns])
        x = np.arange(len(strat.columns))
        strategies = strat.index.tolist()
        colors_s = [PALETTE["accent"], PALETTE["blue"], PALETTE["purple"]]
        w = 0.7 / max(len(strategies), 1)
        for i, s in enumerate(strategies):
            offset = (i - len(strategies)/2) * w + w/2
            ax4.bar(x + offset, strat.loc[s], width=w*0.9,
                    color=colors_s[i % len(colors_s)], alpha=0.85, label=s)
        ax4.set_xticks(x)
        ax4.set_xticklabels(strat.columns, rotation=30, ha="right", fontsize=7)
        ax4.set_title("STRATEGY A vs B vs C (FORMALITY)", loc="left", fontsize=9)
        ax4.legend(framealpha=0, fontsize=8)
        ax4.grid(axis="y", alpha=0.3)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("STYLE TRANSFER BENCHMARK — Results Overview",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99)

    path = out / "charts.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"], dpi=150)
    plt.close()
    print(f"  ✓ {path}  <- README hero")


# ── Entry point ──────────────────────────────────────────────

def generate_all_charts(results_path=None):
    results_path = results_path or RESULTS_DIR / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"No results at {results_path}. Run run_experiment.py first.")

    setup_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_data(results_path)
    print(f"\n Generating charts from {results_path}...")
    print(f"  {len(df)} rows | {df['model'].nunique()} models | {df['style'].nunique()} styles\n")

    chart_formality_heatmap(df, RESULTS_DIR)
    chart_fk_grade(df, RESULTS_DIR)
    chart_strategy_comparison(df, RESULTS_DIR)
    chart_delta_radar(df, RESULTS_DIR)
    chart_compression(df, RESULTS_DIR)
    chart_master(df, RESULTS_DIR)

    print(f"\n All charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    generate_all_charts(Path(args.input) if args.input else None)
