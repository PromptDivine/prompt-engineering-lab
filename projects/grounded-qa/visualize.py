"""
visualize.py
============
Grounded QA — Chart Generator
Project: P5 · prompt-engineering-lab

Charts:
  1. Hallucination rate: ungrounded vs grounded vs cited (per model)
  2. Factual accuracy by condition × model
  3. Unanswerable question handling (refusal rate)
  4. Citation validity (for cited condition)
  5. Hallucination type breakdown (stacked bar)
  6. RAG retrieval accuracy (if rag_results.csv exists)
  7. Master 4-panel hero → results/charts.png
"""

import argparse
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

CONDITION_COLORS = {
    "ungrounded": PALETTE["red"],
    "grounded":   PALETTE["blue"],
    "cited":      PALETTE["green"],
}

HALL_COLORS = {
    "NONE":            PALETTE["green"],
    "FABRICATED_FACT": PALETTE["red"],
    "CONTEXT_LEAK":    PALETTE["orange"],
    "WRONG_SOURCE":    PALETTE["purple"],
    "PARTIAL":         PALETTE["pink"],
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],   "axes.facecolor":  PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],"axes.labelcolor": PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],  "xtick.color":     PALETTE["muted"],
        "ytick.color":      PALETTE["muted"], "text.color":      PALETTE["text"],
        "grid.color":       PALETTE["border"],"grid.linewidth":  0.5,
        "font.family":      "monospace",      "font.size":       9,
        "axes.titlesize":   10,               "axes.titleweight": "bold",
        "axes.titlepad":    10,               "figure.dpi":      150,
    })


def load_data():
    df  = pd.read_csv(RESULTS_DIR / "results.csv")
    df  = df[df["error"].isna() | (df["error"] == "")].copy()
    lb  = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    hr  = pd.read_csv(RESULTS_DIR / "hallucination_report.csv")
    rag = pd.read_csv(RESULTS_DIR / "rag_results.csv") if (RESULTS_DIR / "rag_results.csv").exists() else None
    return df, lb, hr, rag


# ── Chart 1: Hallucination rate by condition ────────────────

def chart_hallucination_rate(df: pd.DataFrame, out: Path):
    hall = df.groupby(["model","condition"])["hallucination_flag"].mean().unstack(fill_value=0)
    condition_order = [c for c in ["ungrounded","grounded","cited"] if c in hall.columns]
    hall = hall[condition_order]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(hall.index))
    n = len(hall.columns)
    width = 0.7 / n

    for i, cond in enumerate(hall.columns):
        offset = (i - n/2) * width + width/2
        color = CONDITION_COLORS.get(cond, PALETTE["blue"])
        bars = ax.bar(x + offset, hall[cond], width=width*0.9, color=color, alpha=0.85, label=cond)
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7, color=PALETTE["muted"])

    ax.set_xticks(x)
    ax.set_xticklabels(hall.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Hallucination Rate (% outputs)")
    ax.set_title("HALLUCINATION RATE — Ungrounded vs Grounded vs Cited", loc="left")
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_hallucination_rate.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 2: Factual accuracy by condition ──────────────────

def chart_factual_accuracy(df: pd.DataFrame, out: Path):
    acc = df.groupby(["model","condition"])["factual_accuracy"].mean().unstack(fill_value=0)
    condition_order = [c for c in ["ungrounded","grounded","cited"] if c in acc.columns]
    acc = acc[condition_order]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(acc.index))
    n = len(acc.columns)
    width = 0.7 / n

    for i, cond in enumerate(acc.columns):
        offset = (i - n/2) * width + width/2
        color = CONDITION_COLORS.get(cond, PALETTE["blue"])
        ax.bar(x + offset, acc[cond], width=width*0.9, color=color, alpha=0.85, label=cond)

    ax.set_xticks(x)
    ax.set_xticklabels(acc.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Factual Accuracy (avg)")
    ax.set_title("FACTUAL ACCURACY — By Condition & Model", loc="left")
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_factual_accuracy.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 3: Unanswerable question handling ─────────────────

def chart_unanswerable(df: pd.DataFrame, out: Path):
    unanswerable = df[df["is_answerable"] == False]
    if unanswerable.empty:
        return

    refusal_rate = unanswerable.groupby(["model","condition"])["unanswerable_correct"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(refusal_rate.index))
    n = len(refusal_rate.columns)
    width = 0.7 / max(n, 1)

    for i, cond in enumerate(refusal_rate.columns):
        offset = (i - n/2) * width + width/2
        color = CONDITION_COLORS.get(cond, PALETTE["blue"])
        ax.bar(x + offset, refusal_rate[cond], width=width*0.9, color=color, alpha=0.85, label=cond)

    ax.set_xticks(x)
    ax.set_xticklabels(refusal_rate.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Correct Refusal Rate")
    ax.set_title("UNANSWERABLE QUESTIONS — Did the model correctly refuse?", loc="left")
    ax.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=0.8, alpha=0.6, label="Perfect")
    ax.legend(framealpha=0, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_unanswerable.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 4: Hallucination type breakdown ───────────────────

def chart_hall_types(hr: pd.DataFrame, out: Path):
    if hr.empty:
        return

    pivot = hr.groupby(["model","hallucination_type"])["count"].sum().unstack(fill_value=0)
    types = [t for t in HALL_COLORS if t in pivot.columns]
    pivot = pivot[types]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))

    for htype in types:
        if htype not in pivot.columns:
            continue
        vals = pivot[htype].values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=HALL_COLORS[htype], alpha=0.85, label=htype, width=0.55)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("HALLUCINATION TYPE BREAKDOWN — Per Model", loc="left")
    ax.legend(framealpha=0, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_hall_types.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Chart 5: Grounding improvement (ungrounded → grounded) ──

def chart_grounding_improvement(df: pd.DataFrame, out: Path):
    conds = df[df["condition"].isin(["ungrounded","grounded"])]
    if conds.empty:
        return

    pivot = conds.groupby(["model","condition"])["hallucination_flag"].mean().unstack(fill_value=0)
    if "ungrounded" not in pivot or "grounded" not in pivot:
        return
    pivot["improvement"] = pivot["ungrounded"] - pivot["grounded"]
    pivot = pivot.sort_values("improvement", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in pivot["improvement"]]
    bars = ax.barh(pivot.index, pivot["improvement"], color=colors, height=0.55, alpha=0.9)

    for bar, val in zip(bars, pivot["improvement"]):
        ax.text(val + 0.005 if val >= 0 else val - 0.005,
                bar.get_y() + bar.get_height()/2,
                f"{val:+.1%}", va="center",
                ha="left" if val >= 0 else "right",
                color=PALETTE["text"], fontsize=8)

    ax.axvline(0, color=PALETTE["muted"], linewidth=0.8)
    ax.set_xlabel("Hallucination Rate Reduction (ungrounded − grounded)")
    ax.set_title("GROUNDING IMPACT — Hallucination reduction per model", loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    path = out / "chart_grounding_improvement.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ {path}")


# ── Master hero chart ────────────────────────────────────────

def chart_master(df: pd.DataFrame, lb: pd.DataFrame, hr: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax3, ax4 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    # Panel 1: Hallucination rate by condition
    hall = df.groupby(["model","condition"])["hallucination_flag"].mean().unstack(fill_value=0)
    cond_order = [c for c in ["ungrounded","grounded","cited"] if c in hall.columns]
    hall = hall[cond_order]
    x = np.arange(len(hall.index))
    n = len(hall.columns)
    w = 0.7 / max(n, 1)
    for i, c in enumerate(hall.columns):
        offset = (i - n/2) * w + w/2
        ax1.bar(x + offset, hall[c], width=w*0.9, color=CONDITION_COLORS.get(c, PALETTE["blue"]), alpha=0.85, label=c)
    ax1.set_xticks(x); ax1.set_xticklabels(hall.index, rotation=15, ha="right", fontsize=7)
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("Hallucination Rate")
    ax1.set_title("HALLUCINATION RATE BY CONDITION", loc="left", fontsize=9)
    ax1.legend(framealpha=0, fontsize=7); ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: Factual accuracy OVERALL per model
    overall = lb[lb["condition"]=="OVERALL"].sort_values("factual_accuracy", ascending=True)
    if not overall.empty and "factual_accuracy" in overall.columns:
        colors2 = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in overall["model"]]
        bars2 = ax2.barh(overall["model"], overall["factual_accuracy"], color=colors2, height=0.55, alpha=0.9)
        for bar, val in zip(bars2, overall["factual_accuracy"]):
            ax2.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.0%}", va="center", fontsize=8, color=PALETTE["text"])
    ax2.set_xlim(0, 1.2); ax2.set_xlabel("Factual Accuracy")
    ax2.set_title("FACTUAL ACCURACY LEADERBOARD", loc="left", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel 3: Grounding improvement
    conds = df[df["condition"].isin(["ungrounded","grounded"])]
    if not conds.empty:
        piv = conds.groupby(["model","condition"])["hallucination_flag"].mean().unstack(fill_value=0)
        if "ungrounded" in piv.columns and "grounded" in piv.columns:
            piv["delta"] = piv["ungrounded"] - piv["grounded"]
            piv = piv.sort_values("delta")
            colors3 = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in piv["delta"]]
            ax3.barh(piv.index, piv["delta"], color=colors3, height=0.55, alpha=0.9)
            ax3.axvline(0, color=PALETTE["muted"], linewidth=0.8)
    ax3.set_xlabel("Hallucination reduction (ungrounded − grounded)")
    ax3.set_title("GROUNDING IMPACT", loc="left", fontsize=9)
    ax3.grid(axis="x", alpha=0.3)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Hallucination type breakdown
    if not hr.empty:
        types = [t for t in HALL_COLORS if t in hr["hallucination_type"].values]
        pivot4 = hr.groupby(["model","hallucination_type"])["count"].sum().unstack(fill_value=0)
        types = [t for t in types if t in pivot4.columns]
        bottom4 = np.zeros(len(pivot4.index))
        x4 = np.arange(len(pivot4.index))
        for htype in types:
            vals = pivot4[htype].values.astype(float)
            ax4.bar(x4, vals, bottom=bottom4, color=HALL_COLORS[htype], alpha=0.85, label=htype, width=0.55)
            bottom4 += vals
        ax4.set_xticks(x4); ax4.set_xticklabels(pivot4.index, rotation=15, ha="right", fontsize=7)
        ax4.set_title("HALLUCINATION TYPES", loc="left", fontsize=9)
        ax4.legend(framealpha=0, fontsize=7); ax4.grid(axis="y", alpha=0.3)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("GROUNDED QA BENCHMARK — Results Overview",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99)
    path = out / "charts.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"], dpi=150)
    plt.close()
    print(f"  ✓ {path}  ← README hero")


def generate_all_charts():
    if not (RESULTS_DIR / "results.csv").exists():
        raise FileNotFoundError("Run run_experiment.py first.")
    setup_style()
    df, lb, hr, rag = load_data()
    print(f"\n Generating charts ({len(df)} rows)...\n")

    chart_hallucination_rate(df, RESULTS_DIR)
    chart_factual_accuracy(df, RESULTS_DIR)
    chart_unanswerable(df, RESULTS_DIR)
    chart_hall_types(hr, RESULTS_DIR)
    chart_grounding_improvement(df, RESULTS_DIR)
    chart_master(df, lb, hr, RESULTS_DIR)
    print(f"\n All charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    generate_all_charts()
