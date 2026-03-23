"""
visualize.py
============
Email Summarizer — Chart Generator
Project: P6 · prompt-engineering-lab

Charts:
  1. Strategy leaderboard (composite score)
  2. Latency vs quality scatter (the key tradeoff chart)
  3. ROUGE by strategy × model heatmap
  4. Tone preservation by tone category
  5. Compression ratio by strategy (how much each condenses)
  6. Word count distribution by strategy
  7. Master 4-panel hero → results/charts.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":      "#0f1117", "surface": "#161820", "border": "#1e2130",
    "text":    "#f0f2f8", "muted":   "#5a6080", "accent": "#e8ff47",
    "blue":    "#47c8ff", "purple":  "#b847ff", "orange": "#ff8c47",
    "green":   "#47ffb2", "red":     "#ff4776", "pink":   "#ff47c8",
}

MODEL_COLORS = {
    "GPT-4o-mini":       PALETTE["blue"],
    "GPT-4o":            PALETTE["accent"],
    "Claude Haiku":      PALETTE["orange"],
    "Claude Sonnet 4.6": PALETTE["purple"],
    "Mistral 7B":        PALETTE["green"],
    "Llama 3 8B":        PALETTE["red"],
}

STRATEGY_COLORS = {
    "tldr":           PALETTE["accent"],
    "bullets":        PALETTE["blue"],
    "formal_paragraph":PALETTE["purple"],
    "casual":         PALETTE["green"],
    "tone_matched":   PALETTE["orange"],
}

STRATEGY_ORDER = ["tldr", "bullets", "formal_paragraph", "casual", "tone_matched"]

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],   "axes.facecolor":  PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],"axes.labelcolor": PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],  "xtick.color":     PALETTE["muted"],
        "ytick.color":      PALETTE["muted"], "text.color":      PALETTE["text"],
        "grid.color":       PALETTE["border"],"grid.linewidth":  0.5,
        "font.family":      "monospace",      "font.size":       9,
        "axes.titlesize":   10,               "axes.titleweight": "bold",
        "figure.dpi":       150,
    })

def load_data():
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    lb = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    lr = pd.read_csv(RESULTS_DIR / "latency_report.csv") if (RESULTS_DIR / "latency_report.csv").exists() else pd.DataFrame()
    return df, lb, lr


# ── Chart 1: Strategy Leaderboard ───────────────────────────

def chart_leaderboard(lb: pd.DataFrame, out: Path):
    metric = "composite" if "composite" in lb.columns else "rouge1"
    strat_avg = lb.groupby("strategy")[metric].mean().reindex(
        [s for s in STRATEGY_ORDER if s in lb["strategy"].values]
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in strat_avg.index]
    bars = ax.barh(strat_avg.index, strat_avg.values, color=colors, height=0.55, alpha=0.9)
    for bar, val in zip(bars, strat_avg.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlim(0, strat_avg.max() * 1.2)
    ax.set_xlabel(f"{metric} (avg across models)")
    ax.set_title("STRATEGY LEADERBOARD", loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out / "chart_leaderboard.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(); print(f"  ✓ {path}")


# ── Chart 2: Latency vs Quality ──────────────────────────────

def chart_latency_quality(df: pd.DataFrame, out: Path):
    model_strat = df.groupby(["model","strategy"])[["latency_s","rouge1","tone_preservation"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in model_strat.iterrows():
        color = MODEL_COLORS.get(row["model"], PALETTE["blue"])
        marker = {"tldr":"o","bullets":"s","formal_paragraph":"D","casual":"^","tone_matched":"*"}.get(row["strategy"],"o")
        ax.scatter(row["latency_s"], row["rouge1"], color=color, s=80,
                   marker=marker, alpha=0.8, zorder=3)

    # Model average diamonds
    for model, grp in df.groupby("model"):
        color = MODEL_COLORS.get(model, PALETTE["blue"])
        ax.scatter(grp["latency_s"].mean(), grp["rouge1"].mean(),
                   color=color, s=200, marker="D", zorder=5,
                   edgecolors="white", linewidths=0.8, label=model)

    # Legend for strategies
    from matplotlib.lines import Line2D
    strategy_handles = [
        Line2D([0],[0], marker=m, color=PALETTE["muted"], linestyle="None",
               markersize=7, label=s)
        for s, m in [("tldr","o"),("bullets","s"),("formal","D"),("casual","^"),("tone_matched","*")]
    ]
    l1 = ax.legend(framealpha=0, fontsize=7, loc="upper left", ncol=2)
    ax.add_artist(l1)
    ax.legend(handles=strategy_handles, framealpha=0, fontsize=7, loc="lower right")

    ax.set_xlabel("Average Latency (s)")
    ax.set_ylabel("ROUGE-1 F1")
    ax.set_title("LATENCY vs QUALITY — Model × Strategy\n(diamonds = model averages)", loc="left")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out / "chart_latency_quality.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(); print(f"  ✓ {path}")


# ── Chart 3: Tone Preservation by Tone ──────────────────────

def chart_tone_preservation(df: pd.DataFrame, out: Path):
    tone_pres = df.groupby(["email_tone","model"])["tone_preservation"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tone_pres.index))
    n = len(tone_pres.columns)
    w = 0.7 / max(n, 1)
    for i, model in enumerate(tone_pres.columns):
        offset = (i - n/2) * w + w/2
        ax.bar(x + offset, tone_pres[model], width=w*0.9,
               color=MODEL_COLORS.get(model, PALETTE["blue"]), alpha=0.85, label=model)
    ax.set_xticks(x); ax.set_xticklabels(tone_pres.index)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Tone Preservation Score")
    ax.set_title("TONE PRESERVATION — By Original Email Tone", loc="left")
    ax.legend(framealpha=0, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out / "chart_tone_preservation.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(); print(f"  ✓ {path}")


# ── Chart 4: Compression by Strategy ────────────────────────

def chart_compression(df: pd.DataFrame, out: Path):
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].values]
    comp = df.groupby("strategy")["compression_ratio"].mean().reindex(strats)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in comp.index]
    bars = ax.bar(comp.index, comp.values, color=colors, alpha=0.85, width=0.55)
    ax.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=1, alpha=0.6, label="Original length")
    for bar, val in zip(bars, comp.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.2f}x", ha="center", fontsize=8, color=PALETTE["text"])
    ax.set_ylabel("Compression Ratio (vs original)")
    ax.set_title("COMPRESSION RATIO — How much each strategy condenses", loc="left")
    ax.legend(framealpha=0, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out / "chart_compression.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(); print(f"  ✓ {path}")


# ── Chart 5: Single vs Thread performance ───────────────────

def chart_single_vs_thread(df: pd.DataFrame, out: Path):
    if "email_type" not in df.columns:
        return
    pivot = df.groupby(["email_type","model"])["rouge1"].mean().unstack(fill_value=0)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(pivot.columns))
    w = 0.35
    types = pivot.index.tolist()
    colors_t = [PALETTE["blue"], PALETTE["orange"]]
    for i, etype in enumerate(types):
        ax.bar(x + (i - len(types)/2) * w + w/2, pivot.loc[etype],
               width=w*0.9, color=colors_t[i % len(colors_t)], alpha=0.85, label=etype)
    ax.set_xticks(x); ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("ROUGE-1 F1")
    ax.set_title("SINGLE vs THREAD EMAILS — ROUGE-1 by Model", loc="left")
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out / "chart_single_vs_thread.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(); print(f"  ✓ {path}")


# ── Master hero chart ────────────────────────────────────────

def chart_master(df: pd.DataFrame, lb: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax3, ax4 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    metric = "composite" if "composite" in lb.columns else "rouge1"

    # Panel 1: Strategy leaderboard
    strat_avg = lb.groupby("strategy")[metric].mean()
    strat_avg = strat_avg.reindex([s for s in STRATEGY_ORDER if s in strat_avg.index]).sort_values()
    colors1 = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in strat_avg.index]
    bars1 = ax1.barh(strat_avg.index, strat_avg.values, color=colors1, height=0.55, alpha=0.9)
    for bar, val in zip(bars1, strat_avg.values):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])
    ax1.set_title("STRATEGY LEADERBOARD", loc="left", fontsize=9)
    ax1.set_xlabel(metric); ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: Latency vs ROUGE scatter
    model_avg = df.groupby("model")[["latency_s","rouge1"]].mean()
    for model, row in model_avg.iterrows():
        color = MODEL_COLORS.get(model, PALETTE["blue"])
        ax2.scatter(row["latency_s"], row["rouge1"], color=color, s=150,
                    marker="D", zorder=4, edgecolors="white", linewidths=0.5)
        ax2.annotate(model, (row["latency_s"], row["rouge1"]),
                     textcoords="offset points", xytext=(6,3),
                     color=color, fontsize=7, fontweight="bold")
    ax2.set_xlabel("Avg Latency (s)"); ax2.set_ylabel("ROUGE-1")
    ax2.set_title("LATENCY vs QUALITY", loc="left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: Tone preservation
    if "tone_preservation" in df.columns:
        tone_pres = df.groupby("email_tone")["tone_preservation"].mean().sort_values()
        colors3 = [PALETTE["blue"]]*len(tone_pres)
        ax3.barh(tone_pres.index, tone_pres.values, color=colors3, height=0.5, alpha=0.85)
        ax3.set_xlim(0, 1.2); ax3.set_xlabel("Tone Preservation Score")
        ax3.set_title("TONE PRESERVATION BY TONE", loc="left", fontsize=9)
        ax3.grid(axis="x", alpha=0.3)
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Compression by strategy
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].values]
    comp = df.groupby("strategy")["compression_ratio"].mean().reindex(strats)
    colors4 = [STRATEGY_COLORS.get(s, PALETTE["blue"]) for s in comp.index]
    ax4.bar(comp.index, comp.values, color=colors4, alpha=0.85, width=0.55)
    ax4.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax4.set_ylabel("Compression Ratio")
    ax4.set_title("COMPRESSION BY STRATEGY", loc="left", fontsize=9)
    ax4.set_xticklabels(comp.index, rotation=20, ha="right", fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("EMAIL SUMMARIZER BENCHMARK — Results Overview",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99)
    path = out / "charts.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"], dpi=150)
    plt.close(); print(f"  ✓ {path}  ← README hero")


def generate_all_charts():
    if not (RESULTS_DIR / "results.csv").exists():
        raise FileNotFoundError("Run run_experiment.py first.")
    setup_style()
    df, lb, lr = load_data()
    print(f"\n Generating charts ({len(df)} rows)...\n")
    chart_leaderboard(lb, RESULTS_DIR)
    chart_latency_quality(df, RESULTS_DIR)
    chart_tone_preservation(df, RESULTS_DIR)
    chart_compression(df, RESULTS_DIR)
    chart_single_vs_thread(df, RESULTS_DIR)
    chart_master(df, lb, RESULTS_DIR)
    print(f"\n All charts saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    generate_all_charts()
