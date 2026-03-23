"""
run_experiment.py
=================
Email Summarizer — Experiment Runner
Project: P6 · prompt-engineering-lab

Usage:
    python run_experiment.py                         # full run
    python run_experiment.py --models openai         # one provider
    python run_experiment.py --strategies tldr,bullets
    python run_experiment.py --quick                 # 5 emails, TLDR only
    python run_experiment.py --tone-match            # run tone-matched prompts only

Outputs:
    results/results.csv       — full results
    results/leaderboard.csv   — ranked by strategy × model
    results/latency_report.csv — latency vs quality tradeoff
"""

import os
import re
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

from evaluation import evaluate_summary, SummaryEval
from tone_detector import detect_tone, select_tone_prompt_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR  = Path("results")
DATA_DIR     = Path("data")
PROMPTS_FILE = Path("prompts/prompts.txt")

MODELS = {
    "openai": {
        "gpt-4o-mini": {"provider": "openai",     "label": "GPT-4o-mini"},
        "gpt-4o":      {"provider": "openai",     "label": "GPT-4o"},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001":    {"provider": "anthropic", "label": "Claude Haiku"},
        "claude-sonnet-4-6": {"provider": "anthropic", "label": "Claude Sonnet 4.6"},
    },
    "openrouter": {
        "google/gemini-2.0-flash-001":  {"provider": "openrouter", "label": "Gemini 2.0 Flash"},
        "meta-llama/llama-3-8b-instruct": {"provider": "openrouter", "label": "Llama 3 8B"},
    },
}


# ── Prompt loading ───────────────────────────────────────────

def load_prompts(path: Path) -> dict:
    prompts = {}
    current_id, current_meta, current_lines, in_template = None, {}, [], False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#") and not in_template:
                continue
            m = re.match(r'^\[(\w+)\]\s+(\S+)', line)
            if m:
                if current_id:
                    prompts[current_id] = {**current_meta, "template": "\n".join(current_lines).strip()}
                current_id, current_meta, current_lines, in_template = m.group(1), {}, [], False
                continue
            if current_id is None:
                continue
            for key in ("strategy", "expected_length", "target_tone"):
                if line.startswith(f"{key}:") and not in_template:
                    current_meta[key] = line.split(":",1)[1].strip()
                    break
            else:
                if line.strip() == "---":
                    in_template = True
                elif in_template:
                    current_lines.append(line)
    if current_id:
        prompts[current_id] = {**current_meta, "template": "\n".join(current_lines).strip()}
    return prompts


def fill_prompt(template: str, email: str, tone: str = "", email_type: str = "") -> str:
    return (template
            .replace("{{email}}", email)
            .replace("{{tone}}", tone)
            .replace("{{type}}", email_type))


# ── Clients ──────────────────────────────────────────────────

def init_clients(model_filter):
    clients, active = {}, set()
    needed = set()
    for grp, grp_models in MODELS.items():
        if model_filter and grp not in model_filter:
            continue
        for _, meta in grp_models.items():
            needed.add(meta["provider"])

    from openai import OpenAI
    import anthropic as ant

    if "openai" in needed and os.environ.get("OPENAI_API_KEY"):
        try:
            clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            active.add("openai"); logger.info("  openai ready")
        except Exception as e:
            logger.warning(f"  openai: {e}")

    if "anthropic" in needed and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            clients["anthropic"] = ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            active.add("anthropic"); logger.info("  anthropic ready")
        except Exception as e:
            logger.warning(f"  anthropic: {e}")

    if "openrouter" in needed and os.environ.get("OPENROUTER_API_KEY"):
        try:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            active.add("openrouter"); logger.info("  openrouter ready")
        except Exception as e:
            logger.warning(f"  openrouter: {e}")

    return clients, active


def call_model(provider, model_id, prompt, clients):
    for attempt in range(3):
        try:
            t0 = time.time()
            if provider in ("openai", "openrouter"):
                resp = clients[provider].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=300,
                )
                return resp.choices[0].message.content.strip(), time.time() - t0
            elif provider == "anthropic":
                resp = clients[provider].messages.create(
                    model=model_id, max_tokens=300, temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip(), time.time() - t0
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)


# ── Build reports ────────────────────────────────────────────

def build_leaderboard(df: pd.DataFrame):
    df_clean = df[df["error"].isna() | (df["error"] == "")]
    metrics  = ["rouge1", "rouge2", "rougeL", "tone_preservation",
                "compression_ratio", "latency_s", "word_count"]
    available = [m for m in metrics if m in df_clean.columns]

    lb = df_clean.groupby(["model", "strategy"])[available].mean().round(4).reset_index()

    # Composite: ROUGE + tone preservation
    if all(c in lb.columns for c in ["rouge1", "rougeL", "tone_preservation"]):
        lb["composite"] = (
            0.3 * lb["rouge1"] + 0.3 * lb["rougeL"] + 0.4 * lb["tone_preservation"]
        ).round(4)
        lb = lb.sort_values("composite", ascending=False)
    else:
        lb = lb.sort_values("rouge1", ascending=False)

    lb.insert(0, "rank", range(1, len(lb)+1))
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    logger.info("  leaderboard.csv saved")


def build_latency_report(df: pd.DataFrame):
    df_clean = df[df["error"].isna() | (df["error"] == "")]
    if "rouge1" not in df_clean.columns:
        return

    lr = df_clean.groupby(["model", "strategy"])[["latency_s", "rouge1", "tone_preservation"]].mean().round(4).reset_index()
    lr["quality_per_second"] = (lr["rouge1"] / lr["latency_s"]).round(4)
    lr = lr.sort_values("quality_per_second", ascending=False)
    lr.to_csv(RESULTS_DIR / "latency_report.csv", index=False)
    logger.info("  latency_report.csv saved")


# ── Main experiment loop ─────────────────────────────────────

def run_experiment(
    model_filter=None,
    strategy_filter=None,
    email_filter=None,
    tone_match_only=False,
    quick=False,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    emails  = pd.read_csv(DATA_DIR / "emails.csv")
    prompts = load_prompts(PROMPTS_FILE)

    if quick:
        emails = emails.head(5)
        strategy_filter = ["tldr"]
        logger.info("Quick mode: 5 emails, TLDR only")

    if email_filter:
        emails = emails[emails["email_id"].isin(email_filter)]

    if tone_match_only:
        prompts = {k: v for k, v in prompts.items() if v.get("strategy") == "tone_matched"}
    elif strategy_filter:
        prompts = {k: v for k, v in prompts.items()
                   if v.get("strategy") in strategy_filter
                   or v.get("strategy") == "tone_matched"}

    clients, active = init_clients(model_filter)
    if not clients:
        raise RuntimeError("No API clients initialized.")

    all_results = []
    total = errors = 0

    for _, email in emails.iterrows():
        # Detect tone once per email
        tone_result = detect_tone(email["body"])
        detected_tone = tone_result.primary_tone

        for pid, pmeta in prompts.items():
            # For tone-matched prompts, only run the one matching this email's tone
            if pmeta.get("strategy") == "tone_matched":
                expected_tone_prompt = select_tone_prompt_id(detected_tone)
                if pid != expected_tone_prompt:
                    continue

            filled = fill_prompt(
                pmeta["template"],
                email["body"],
                tone=detected_tone,
                email_type=email["type"],
            )

            for grp, grp_models in MODELS.items():
                if model_filter and grp not in model_filter:
                    continue
                for model_id, mmeta in grp_models.items():
                    provider = mmeta["provider"]
                    if provider not in active:
                        continue

                    logger.info(f"  [{email['email_id']}] {pid} ({pmeta['strategy']}) → {mmeta['label']}")

                    try:
                        output, latency = call_model(provider, model_id, filled, clients)
                        result = evaluate_summary(
                            email_id=email["email_id"],
                            email_body=email["body"],
                            email_tone=detected_tone,
                            email_type=email["type"],
                            reference_summary=email["reference_summary"],
                            model=mmeta["label"],
                            prompt_id=pid,
                            strategy=pmeta["strategy"],
                            summary=output,
                            latency_s=latency,
                        )
                        total += 1
                    except Exception as e:
                        logger.error(f"    FAILED: {e}")
                        result = SummaryEval(
                            email_id=email["email_id"], email_type=email["type"],
                            email_tone=detected_tone, model=mmeta["label"],
                            prompt_id=pid, strategy=pmeta["strategy"],
                            summary="", latency_s=0.0, error=str(e),
                        )
                        errors += 1

                    all_results.append(result.to_dict())
                    time.sleep(0.3)

    if not all_results:
        raise RuntimeError("No results generated.")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    logger.info(f"\n  results.csv saved ({total} runs, {errors} errors)")

    build_leaderboard(df)
    build_latency_report(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     type=str)
    parser.add_argument("--strategies", type=str)
    parser.add_argument("--emails",     type=str)
    parser.add_argument("--tone-match", action="store_true")
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_filter    = args.models.split(",")     if args.models     else None,
        strategy_filter = args.strategies.split(",") if args.strategies else None,
        email_filter    = args.emails.split(",")     if args.emails     else None,
        tone_match_only = args.tone_match,
        quick           = args.quick,
    )
