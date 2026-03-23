"""
run_experiment.py
=================
Grounded QA — Experiment Runner
Project: P5 · prompt-engineering-lab

Two modes:
  benchmark  — static context dataset, 3 conditions (ungrounded/grounded/cited)
  rag        — live RAG pipeline over data/documents/

Usage:
    python run_experiment.py                       # full static benchmark
    python run_experiment.py --mode rag            # RAG demo
    python run_experiment.py --mode both           # benchmark + RAG
    python run_experiment.py --models openai       # single provider
    python run_experiment.py --quick               # 6 questions only
    python run_experiment.py --conditions grounded # one condition

Outputs:
    results/results.csv          — all eval results
    results/leaderboard.csv      — model rankings
    results/hallucination_report.csv — hallucination breakdown
    results/rag_results.csv      — RAG pipeline results (if --mode rag/both)
"""

import os
import re
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

from evaluation import evaluate_qa, QAEvalResult

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
        "mistralai/mistral-small-creative":  {"provider": "openrouter", "label": "Mistral small creative"},
        "meta-llama/llama-3-8b-instruct": {"provider": "openrouter", "label": "Llama 3 8B"},
    },
}

CONDITION_PROMPT_MAP = {
    "ungrounded": ["UG01"],
    "grounded":   ["GR01", "GR02", "GR03"],
    "cited":      ["CT01", "CT02", "CT03"],
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
            if line.startswith("condition:") and not in_template:
                current_meta["condition"] = line.split(":", 1)[1].strip()
            elif line.startswith("strategy:") and not in_template:
                current_meta["strategy"] = line.split(":", 1)[1].strip()
            elif line.strip() == "---":
                in_template = True
            elif in_template:
                current_lines.append(line)

    if current_id:
        prompts[current_id] = {**current_meta, "template": "\n".join(current_lines).strip()}
    return prompts


def fill_prompt(template: str, question: str, context: str = "") -> str:
    return template.replace("{{question}}", question).replace("{{context}}", context)


# ── Clients ──────────────────────────────────────────────────

def init_clients(model_filter, models_config):
    clients = {}
    active = set()
    needed = set()
    for grp, grp_models in models_config.items():
        if model_filter and grp not in model_filter:
            continue
        for _, meta in grp_models.items():
            needed.add(meta["provider"])

    def try_init(provider, env_key, factory):
        if provider in needed and os.environ.get(env_key):
            try:
                clients[provider] = factory()
                active.add(provider)
                logger.info(f"  Client ready: {provider}")
            except Exception as e:
                logger.warning(f"  {provider} init failed: {e}")

    from openai import OpenAI
    import anthropic as ant

    try_init("openai",     "OPENAI_API_KEY",     lambda: OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    try_init("anthropic",  "ANTHROPIC_API_KEY",  lambda: ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    try_init("openrouter", "OPENROUTER_API_KEY", lambda: OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    ))
    return clients, active


def call_model(provider, model_id, prompt, clients):
    for attempt in range(3):
        try:
            t0 = time.time()
            if provider in ("openai", "openrouter"):
                resp = clients[provider].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip(), time.time() - t0
            elif provider == "anthropic":
                resp = clients[provider].messages.create(
                    model=model_id, max_tokens=400, temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip(), time.time() - t0
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Attempt {attempt+1} failed: {e} — retrying")
            time.sleep(2)


# ── Static benchmark ─────────────────────────────────────────

def run_benchmark(
    model_filter=None,
    condition_filter=None,
    question_filter=None,
    quick=False,
    clients=None,
    active_providers=None,
    models_config=None,
) -> pd.DataFrame:

    questions = pd.read_csv(DATA_DIR / "contexts.csv")
    prompts   = load_prompts(PROMPTS_FILE)

    if quick:
        questions = questions.head(6)
        condition_filter = condition_filter or ["grounded"]
        logger.info("Quick mode: 6 questions, grounded only")

    if question_filter:
        questions = questions[questions["question_id"].isin(question_filter)]
    if condition_filter:
        prompts = {k: v for k, v in prompts.items()
                   if v.get("condition") in condition_filter}

    all_results = []
    total = errors = 0

    for _, q in questions.iterrows():
        for pid, pmeta in prompts.items():
            condition = pmeta.get("condition", "grounded")
            context   = q["context"] if condition != "ungrounded" else ""
            filled    = fill_prompt(pmeta["template"], q["question"], context)

            for grp, grp_models in models_config.items():
                if model_filter and grp not in model_filter:
                    continue
                for model_id, mmeta in grp_models.items():
                    provider = mmeta["provider"]
                    if provider not in active_providers:
                        continue

                    logger.info(f"  [{q['question_id']}] {pid} ({condition}) → {mmeta['label']}")

                    try:
                        output, latency = call_model(provider, model_id, filled, clients)
                        result = evaluate_qa(
                            question_id=q["question_id"],
                            condition=condition,
                            prompt_id=pid,
                            model=mmeta["label"],
                            question=q["question"],
                            context=context,
                            ground_truth=q["ground_truth_answer"],
                            is_answerable=bool(q["is_answerable"]),
                            output=output,
                            latency_s=latency,
                        )
                        total += 1
                    except Exception as e:
                        logger.error(f"    FAILED: {e}")
                        result = QAEvalResult(
                            question_id=q["question_id"], condition=condition,
                            prompt_id=pid, model=mmeta["label"],
                            question=q["question"], context=context,
                            ground_truth=q["ground_truth_answer"],
                            is_answerable=bool(q["is_answerable"]),
                            output="", latency_s=0.0, error=str(e),
                        )
                        errors += 1

                    all_results.append(result.to_dict())
                    time.sleep(0.3)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    logger.info(f"\n  results.csv saved ({total} runs, {errors} errors)")

    _build_leaderboard(df)
    _build_hallucination_report(df)
    return df


def _build_leaderboard(df: pd.DataFrame):
    df_clean = df[df["error"].isna() | (df["error"] == "")]
    metrics = ["factual_accuracy", "grounding_rate", "hallucination_flag", "latency_s"]
    available = [m for m in metrics if m in df_clean.columns]

    lb = df_clean.groupby(["model", "condition"])[available].mean().round(4).reset_index()

    # Overall
    overall = df_clean.groupby("model")[available].mean().round(4).reset_index()
    overall["condition"] = "OVERALL"
    lb = pd.concat([lb, overall], ignore_index=True)

    # Hallucination rate = percentage of outputs that hallucinated
    if "hallucination_flag" in lb.columns:
        lb["hallucination_rate"] = lb["hallucination_flag"].round(3)

    lb = lb.sort_values(["condition","factual_accuracy"], ascending=[True, False])
    lb.insert(0, "rank", range(1, len(lb)+1))
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    logger.info("  leaderboard.csv saved")


def _build_hallucination_report(df: pd.DataFrame):
    df_clean = df[df["error"].isna() | (df["error"] == "")]
    if "hallucination_type" not in df_clean.columns:
        return

    report = (
        df_clean.groupby(["model", "condition", "hallucination_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )
    total_per_model = df_clean.groupby("model").size().reset_index(name="total")
    report = report.merge(total_per_model, on="model")
    report["pct"] = (report["count"] / report["total"] * 100).round(1)
    report.to_csv(RESULTS_DIR / "hallucination_report.csv", index=False)
    logger.info("  hallucination_report.csv saved")


# ── RAG pipeline ─────────────────────────────────────────────

RAG_QUESTIONS = [
    {"id": "R01", "question": "What are the penalties for violating the EU AI Act?",
     "expected_source": "eu_ai_act"},
    {"id": "R02", "question": "How does the energy density of sodium-ion batteries compare to lithium-ion?",
     "expected_source": "sodium_battery"},
    {"id": "R03", "question": "How many hectares of forest burned in Canada during 2023?",
     "expected_source": "climate_2023"},
    {"id": "R04", "question": "What computational threshold triggers extra obligations under the EU AI Act?",
     "expected_source": "eu_ai_act"},
    {"id": "R05", "question": "Who leads the MIT sodium battery research team?",
     "expected_source": "sodium_battery"},
    {"id": "R06", "question": "What was the average sea surface temperature anomaly in 2023?",
     "expected_source": "climate_2023"},
]

RAG_PROMPT = """Answer the question using ONLY the retrieved context below.
If the answer is not in the context, say: "Not found in retrieved documents."
Always cite which source your answer comes from.

Retrieved Context:
{context}

Question: {question}

Answer (with source citation):"""


def run_rag(
    model_filter=None,
    clients=None,
    active_providers=None,
    models_config=None,
) -> pd.DataFrame:

    from retriever import Retriever

    logger.info("\n  Building RAG index...")
    retriever = Retriever(docs_dir=str(DATA_DIR / "documents"))
    retriever.index()

    all_results = []

    for q in RAG_QUESTIONS:
        context = retriever.retrieve_as_context(q["question"], top_k=3)
        retrieved_sources = [c.doc_name for c in retriever.retrieve(q["question"], top_k=3)]
        source_correct = q["expected_source"] in retrieved_sources

        filled = RAG_PROMPT.format(context=context, question=q["question"])

        for grp, grp_models in models_config.items():
            if model_filter and grp not in model_filter:
                continue
            for model_id, mmeta in grp_models.items():
                provider = mmeta["provider"]
                if provider not in active_providers:
                    continue

                logger.info(f"  [RAG {q['id']}] → {mmeta['label']}")
                try:
                    output, latency = call_model(provider, model_id, filled, clients)
                except Exception as e:
                    output, latency = "", 0.0
                    logger.error(f"    FAILED: {e}")

                all_results.append({
                    "question_id":      q["id"],
                    "question":         q["question"],
                    "model":            mmeta["label"],
                    "retrieved_sources":str(retrieved_sources),
                    "source_correct":   source_correct,
                    "expected_source":  q["expected_source"],
                    "output":           output,
                    "latency_s":        round(latency, 3),
                })
                time.sleep(0.3)

    df_rag = pd.DataFrame(all_results)
    df_rag.to_csv(RESULTS_DIR / "rag_results.csv", index=False)
    logger.info("  rag_results.csv saved")

    retrieval_accuracy = df_rag["source_correct"].mean()
    logger.info(f"  Retrieval accuracy: {retrieval_accuracy:.1%}")
    return df_rag


# ── CLI ──────────────────────────────────────────────────────

def run_experiment(
    mode="benchmark",
    model_filter=None,
    condition_filter=None,
    question_filter=None,
    quick=False,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    clients, active = init_clients(model_filter, MODELS)
    if not clients:
        raise RuntimeError("No API clients initialized.")

    dfs = {}
    if mode in ("benchmark", "both"):
        dfs["benchmark"] = run_benchmark(
            model_filter=model_filter,
            condition_filter=condition_filter,
            question_filter=question_filter,
            quick=quick,
            clients=clients,
            active_providers=active,
            models_config=MODELS,
        )

    if mode in ("rag", "both"):
        dfs["rag"] = run_rag(
            model_filter=model_filter,
            clients=clients,
            active_providers=active,
            models_config=MODELS,
        )

    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="benchmark", choices=["benchmark","rag","both"])
    parser.add_argument("--models",     type=str)
    parser.add_argument("--conditions", type=str)
    parser.add_argument("--questions",  type=str)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    run_experiment(
        mode            = args.mode,
        model_filter    = args.models.split(",")     if args.models     else None,
        condition_filter= args.conditions.split(",") if args.conditions else None,
        question_filter = args.questions.split(",")  if args.questions  else None,
        quick           = args.quick,
    )
