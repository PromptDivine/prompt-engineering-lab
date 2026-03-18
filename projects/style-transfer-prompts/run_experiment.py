"""
run_experiment.py
=================
Style Transfer Prompts — Experiment Runner
"""

import os
import re
import csv
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from evaluation import compute_deltas, TransferResult  # removed duplicate + unused imports
from evaluation import evaluate_transfer as ev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")
PROMPTS_FILE = Path("prompts/prompts.txt")

# ── Models ──────────────────────────────────────────────────

MODELS = {
    "openai": {
        "gpt-4o-mini": {"provider": "openai",     "label": "GPT-4o-mini"},
        "gpt-4o":      {"provider": "openai",     "label": "GPT-4o"},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {"provider": "anthropic", "label": "Claude Haiku"},
        "claude-sonnet-4-6":         {"provider": "anthropic", "label": "Claude Sonnet 4.6"},
    },
    "openrouter": {
        "mistralai/mistral-small-creative":  {"provider": "openrouter", "label": "Mistral small creative"},
        "meta-llama/llama-3-8b-instruct": {"provider": "openrouter", "label": "Llama 3 8B"},
    },
}

# ── Prompt loading ───────────────────────────────────────────

def load_prompts(path: Path) -> dict:
    prompts = {}
    current_id = None
    current_meta = {}
    current_lines = []
    in_template = False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            # Check for section header FIRST, before skipping comments
            m = re.match(r'^#{0,3}\s*\[(\w+)\]\s+(\S+)', line)
            if m:
                if current_id:
                    prompts[current_id] = {**current_meta, "template": "\n".join(current_lines).strip()}
                current_id = m.group(1)
                current_meta = {}
                current_lines = []
                in_template = False
                continue

            # NOW skip pure comments
            if line.startswith("#") and not in_template:
                continue

            if current_id is None:
                continue

            for key in ("style", "strategy", "formality_target"):
                if line.startswith(f"{key}:") and not in_template:
                    current_meta[key] = line.split(":", 1)[1].strip()
                    break
            else:
                if (
                    not any(line.startswith(f"{k}:") for k in ("style", "strategy", "formality_target"))
                    and line.strip() != ""
                    and not in_template
                ):
                    in_template = True
                    current_lines.append(line)
                elif in_template:
                    current_lines.append(line)

    if current_id:
        prompts[current_id] = {**current_meta, "template": "\n".join(current_lines).strip()}

    return prompts

# ── API clients ──────────────────────────────────────────────

def get_openai_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=key)

def get_anthropic_client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key)

def get_openrouter_client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )


def call_model(provider: str, model_id: str, prompt: str, clients: dict) -> tuple:
    last_error = None
    for attempt in range(3):
        try:
            t0 = time.time()

            if provider == "openai":
                resp = clients["openai"].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=600,
                )
                text = resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                resp = clients["anthropic"].messages.create(
                    model=model_id,
                    max_tokens=600,
                    temperature=0.4,
                    messages=[{"role": "user", "content": prompt}],
                )
                # safer parsing
                text = "".join(
                    block.text for block in resp.content if hasattr(block, "text")
                ).strip()

            elif provider == "openrouter":
                resp = clients["openrouter"].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=600,
                )
                text = resp.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unknown provider: {provider}")

            return text, time.time() - t0

        except Exception as e:
            last_error = e
            if attempt == 2:
                raise last_error
            logger.warning(f"Attempt {attempt+1} failed: {e} — retrying in 2s")
            time.sleep(2)


# ── Core experiment loop ─────────────────────────────────────

def fill_prompt(template: str, text: str) -> str:
    return template.replace("{{TEXT}}", text)

def run_experiment(
    model_filter=None,
    style_filter=None,
    text_filter=None,
    run_llm_judge=False,
    quick=False,
) -> pd.DataFrame:

    RESULTS_DIR.mkdir(exist_ok=True)

    texts   = pd.read_csv(DATA_DIR / "source_texts.csv")
    prompts = load_prompts(PROMPTS_FILE)

    if quick:
        texts = texts.head(2)
        prompts = {k: v for k, v in prompts.items() if k.endswith("A")}
        if not style_filter:
            style_filter = ["journalism", "academic", "casual"]
        logger.info("Quick mode: 2 texts, strategy A only, 3 styles")

    if text_filter:
        texts = texts[texts["id"].isin(text_filter)]

    if style_filter:
        prompts = {k: v for k, v in prompts.items() if v.get("style") in style_filter}

    # Init clients
    clients = {}
    active_providers = set()

    needed = set()
    for group, group_models in MODELS.items():
        if model_filter and group not in model_filter:
            continue
        for _, meta in group_models.items():
            needed.add(meta["provider"])

    init_map = {
        "openai":     ("OPENAI_API_KEY",     get_openai_client),
        "anthropic":  ("ANTHROPIC_API_KEY",  get_anthropic_client),
        "openrouter": ("OPENROUTER_API_KEY", get_openrouter_client),
    }

    for provider, (env_key, factory) in init_map.items():
        if provider in needed and os.environ.get(env_key):
            try:
                clients[provider] = factory()
                active_providers.add(provider)
                logger.info(f"  Client initialized: {provider}")
            except Exception as e:
                logger.warning(f"  {provider} init failed: {e}")

    if not clients:
        raise RuntimeError("No API clients initialized. Check your environment variables.")

    judge_client = clients.get("openai") if run_llm_judge else None

    source_metrics_cache = {}

    all_results = []
    total = 0
    errors = 0

    for _, text_row in texts.iterrows():
        src_key = text_row["id"]

        if src_key not in source_metrics_cache:
            source_metrics_cache[src_key] = ev(
                str(text_row["text"] or ""),
                str(text_row["text"] or ""),
                "source"
            )

        for pid, pmeta in prompts.items():
            for group, group_models in MODELS.items():
                if model_filter and group not in model_filter:
                    continue
                for model_id, mmeta in group_models.items():
                    provider = mmeta["provider"]
                    if provider not in active_providers:
                        continue

                    logger.info(f"  [{text_row['id']}] {pid} ({pmeta.get('style')}) → {mmeta['label']}")

                    source_text = str(text_row["text"] or "")
                    filled = fill_prompt(pmeta["template"], source_text)

                    try:
                        output, latency = call_model(provider, model_id, filled, clients)

                        metrics = ev(
                            source=source_text,
                            output=output,
                            style=pmeta.get("style", ""),
                            judge_client=judge_client,
                            run_llm_judge=run_llm_judge,
                        )

                        deltas = compute_deltas(source_metrics_cache[src_key], metrics)

                        result = TransferResult(
                            source_id=text_row["id"],
                            source_domain=text_row["domain"],
                            model=mmeta["label"],
                            prompt_id=pid,
                            style=pmeta.get("style", ""),
                            strategy=pmeta.get("strategy", ""),
                            output=output,
                            latency_s=latency,
                            metrics=metrics,
                        )

                        row = result.to_dict()
                        row.update(deltas)
                        all_results.append(row)
                        total += 1

                    except Exception as e:
                        logger.error(f"    FAILED: {e}")
                        all_results.append({
                            "source_id": text_row["id"],
                            "source_domain": text_row["domain"],
                            "model": mmeta["label"],
                            "prompt_id": pid,
                            "style": pmeta.get("style",""),
                            "strategy": pmeta.get("strategy",""),
                            "output": "",
                            "latency_s": 0.0,
                            "error": str(e),
                        })
                        errors += 1

                    time.sleep(0.3)
    
    logger.info(f"Texts loaded: {len(texts)}")
    logger.info(f"Prompts loaded: {len(prompts)}")
    logger.info(f"Active providers: {active_providers}")
    logger.info(f"Total results: {len(all_results)}")
    if not all_results:
        raise RuntimeError("No results generated.")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    logger.info(f"\n  Saved results.csv  ({total} runs, {errors} errors)")

    # Leaderboard
    metric_cols = ["fk_grade", "formality_score", "unique_word_ratio",
                   "sentiment_polarity", "compression_ratio", "latency_s"]

    if run_llm_judge:
        metric_cols += ["judge_style_adherence", "judge_fluency", "judge_meaning_preserved", "judge_overall"]

    available = [c for c in metric_cols if c in df.columns]

    # FIXED error filtering
    if "error" in df.columns:
        df_clean = df[df["error"].isna() | (df["error"] == "")]
    else:
        df_clean = df

    lb = (
        df_clean
        .groupby(["model", "style", "strategy", "prompt_id"])[available]
        .mean()
        .round(4)
        .reset_index()
    )

    if run_llm_judge and "judge_overall" in lb.columns:
        lb = lb.sort_values("judge_overall", ascending=False)
    else:
        lb = lb.sort_values("formality_score", ascending=False)

    lb.insert(0, "rank", range(1, len(lb)+1))
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    logger.info(f"  Saved leaderboard.csv")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    type=str)
    parser.add_argument("--styles",    type=str)
    parser.add_argument("--texts",     type=str)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--quick",     action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_filter  = args.models.split(",")  if args.models  else None,
        style_filter  = args.styles.split(",")  if args.styles  else None,
        text_filter   = args.texts.split(",")   if args.texts   else None,
        run_llm_judge = args.llm_judge,
        quick         = args.quick,
    )
