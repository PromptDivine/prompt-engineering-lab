"""
update_findings.py
==================
Email Summarizer — Auto-populate findings
Project: P6 · prompt-engineering-lab

Run AFTER run_experiment.py.

Usage:
    python update_findings.py
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")
STRATEGY_ORDER = ["tldr", "bullets", "formal_paragraph", "casual", "tone_matched"]


def load():
    df  = pd.read_csv(RESULTS_DIR / "results.csv")
    lb  = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    lr  = pd.read_csv(RESULTS_DIR / "latency_report.csv") if (RESULTS_DIR / "latency_report.csv").exists() else pd.DataFrame()
    df  = df[df["error"].isna() | (df["error"] == "")].copy()
    return df, lb, lr


def build_leaderboard_table(lb: pd.DataFrame) -> str:
    metric = "composite" if "composite" in lb.columns else "rouge1"
    model_lb = lb.groupby("model")[[metric, "tone_preservation", "latency_s"]].mean().round(3)
    model_lb = model_lb.sort_values(metric, ascending=False).reset_index()

    rows = [
        "| Rank | Model | Composite Score | Tone Preservation | Avg Latency |",
        "|------|-------|----------------|-------------------|-------------|",
    ]
    for rank, (_, row) in enumerate(model_lb.iterrows(), 1):
        rows.append(
            f"| {rank} | {row['model']} "
            f"| {row[metric]:.3f} "
            f"| {row.get('tone_preservation', 0):.3f} "
            f"| {row.get('latency_s', 0):.2f}s |"
        )
    return "\n".join(rows)


def build_strategy_table(lb: pd.DataFrame) -> str:
    metric = "composite" if "composite" in lb.columns else "rouge1"
    strat_lb = lb.groupby("strategy")[[metric, "compression_ratio", "word_count"]].mean().round(3)
    strat_lb = strat_lb.reindex([s for s in STRATEGY_ORDER if s in strat_lb.index])

    rows = [
        "| Strategy | Score | Avg Compression | Avg Word Count |",
        "|----------|-------|----------------|----------------|",
    ]
    for strat, row in strat_lb.iterrows():
        rows.append(
            f"| {strat} | {row[metric]:.3f} "
            f"| {row.get('compression_ratio', 0):.2f}x "
            f"| {int(row.get('word_count', 0))} |"
        )
    return "\n".join(rows)


def update_readme(lb_table: str, strat_table: str):
    content = README_PATH.read_text(encoding="utf-8")

    for start_marker, end_marker, replacement in [
        ("| Rank | Model | Composite Score |", "\n\n*Run", lb_table),
        ("| Strategy | Score |", "\n\n---", strat_table),
    ]:
        si = content.find(start_marker)
        ei = content.find(end_marker, si)
        if si != -1 and ei != -1:
            content = content[:si] + replacement + content[ei:]

    README_PATH.write_text(content, encoding="utf-8")
    print("  README.md updated.")


def print_notebook_findings(df: pd.DataFrame, lb: pd.DataFrame, lr: pd.DataFrame):
    metric = "composite" if "composite" in lb.columns else "rouge1"

    model_avg  = lb.groupby("model")[metric].mean().sort_values(ascending=False)
    best_model = model_avg.index[0]; best_score = model_avg.iloc[0]

    strat_avg  = lb.groupby("strategy")[metric].mean()
    strat_avail = strat_avg.reindex([s for s in STRATEGY_ORDER if s in strat_avg.index])
    best_strat  = strat_avail.idxmax(); best_strat_score = strat_avail.max()

    tone_pres = df.groupby("email_tone")["tone_preservation"].mean().sort_values(ascending=False)
    best_tone = tone_pres.index[0]; best_tone_score = tone_pres.iloc[0]
    worst_tone = tone_pres.index[-1]; worst_tone_score = tone_pres.iloc[-1]

    comp = df.groupby("strategy")["compression_ratio"].mean()
    most_compressed = comp.idxmin(); comp_ratio = comp.min()

    lat = df.groupby("model")["latency_s"].mean().sort_values()
    fastest = lat.index[0]; fastest_lat = lat.iloc[0]
    slowest = lat.index[-1]; slowest_lat = lat.iloc[-1]

    best_qps = ""
    if not lr.empty and "quality_per_second" in lr.columns:
        top = lr.sort_values("quality_per_second", ascending=False).iloc[0]
        best_qps = f"| `{top['model']}` × `{top['strategy']}` — {top['quality_per_second']:.3f} ROUGE/second"

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Best model overall:** `{best_model}` — composite score {best_score:.3f}
2. **Best strategy:** `{best_strat}` — avg score {best_strat_score:.3f}
3. **Tone preservation:** best on `{best_tone}` emails ({best_tone_score:.3f}), lowest on `{worst_tone}` ({worst_tone_score:.3f})
4. **Most aggressive compression:** `{most_compressed}` strategy — {comp_ratio:.2f}x original length
5. **Fastest model:** `{fastest}` ({fastest_lat:.2f}s avg) | Slowest: `{slowest}` ({slowest_lat:.2f}s avg)
6. **Best quality/latency:** {best_qps if best_qps else "see results/latency_report.csv"}
7. **Key insight:** [Fill in after reviewing chart_latency_quality.png]

---
*Demo: `python app.py` → http://127.0.0.1:7860*
""")


def main():
    print("Loading results...")
    df, lb, lr = load()
    print(f"  {len(df)} rows | {df['model'].nunique()} models | {df['strategy'].nunique()} strategies\n")
    print("Building README tables...")
    update_readme(build_leaderboard_table(lb), build_strategy_table(lb))
    print_notebook_findings(df, lb, lr)
    print("\nDone.")


if __name__ == "__main__":
    main()
