"""
cli.py
======
promptlab CLI — command-line interface
Project: P4 · prompt-engineering-lab

Commands:
    promptlab run         Run a batch evaluation from a YAML/JSON test file
    promptlab ab          A/B compare two prompts from the command line
    promptlab regression  Save or check a regression baseline
    promptlab list        List saved baselines

Usage examples:
    promptlab run tests/summarizer.yaml
    promptlab run tests/summarizer.yaml --models gpt-4o-mini --output results/
    promptlab ab --prompt-a "Summarize: {{text}}" --prompt-b "TL;DR: {{text}}" --inputs tests/inputs.json
    promptlab regression save --name summarizer_v1 --test tests/summarizer.yaml
    promptlab regression check --name summarizer_v1 --test tests/summarizer.yaml
    promptlab list
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Test file loader ─────────────────────────────────────────

def load_test_file(path: str) -> dict:
    """
    Load a YAML or JSON test definition file.

    YAML format:
        prompts:
          v1: "Summarize: {{text}}"
          v2: "TL;DR: {{text}}"
        inputs:
          - id: doc1
            text: "..."
        models:
          - gpt-4o-mini
          - claude-haiku-4-5-20251001
        checks:
          - type: word_limit
            n: 100
          - type: must_not_contain
            phrase: "I cannot"
        llm_judge: false
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for .yaml files: pip install pyyaml")
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix} (use .yaml or .json)")


def build_checks_from_config(checks_config: list):
    """Convert check definitions from config file to RubricCheck objects."""
    from promptlab.scorers import RubricScorer
    from promptlab.lab import PromptLab

    check_map = {
        "word_limit":        lambda c: PromptLab.word_limit(c["n"]),
        "word_minimum":      lambda c: PromptLab.word_minimum(c["n"]),
        "must_contain":      lambda c: PromptLab.must_contain(c["phrase"]),
        "must_not_contain":  lambda c: PromptLab.must_not_contain(c["phrase"]),
        "no_refusal":        lambda c: PromptLab.no_refusal(),
        "json_valid":        lambda c: PromptLab.json_valid(),
        "numbered_list":     lambda c: PromptLab.numbered_list(c.get("n", 1)),
        "contains_pattern":  lambda c: PromptLab.contains_pattern(c["pattern"], c.get("name","pattern")),
    }

    checks = []
    for c in checks_config:
        ctype = c.get("type")
        if ctype in check_map:
            checks.append(check_map[ctype](c))
        else:
            logger.warning(f"Unknown check type: {ctype}")
    return checks


# ── Commands ─────────────────────────────────────────────────

def cmd_run(args):
    """Run a batch evaluation."""
    from promptlab.lab import PromptLab

    config = load_test_file(args.test)

    models = args.models.split(",") if args.models else config.get("models", ["gpt-4o-mini"])
    prompts = config.get("prompts", {})
    inputs  = config.get("inputs", [])
    checks_config = config.get("checks", [])
    llm_judge = args.llm_judge or config.get("llm_judge", False)

    if not prompts:
        logger.error("No prompts defined in test file.")
        sys.exit(1)

    checks = build_checks_from_config(checks_config) if checks_config else None

    lab = PromptLab(models=models)
    batch = lab.run(
        prompts=prompts,
        inputs=inputs,
        checks=checks,
        llm_judge=llm_judge,
        run_id=args.run_id or Path(args.test).stem,
    )

    batch.report.print_summary()

    output_dir = args.output or "results"
    batch.save(results_dir=output_dir)
    batch.plot(output_path=f"{output_dir}/{batch.run_id}_chart.png")
    logger.info(f"\n  Results saved to {output_dir}/")


def cmd_ab(args):
    """A/B comparison between two prompts."""
    from promptlab.lab import PromptLab

    models = args.models.split(",") if args.models else ["gpt-4o-mini"]

    # Load inputs
    if args.inputs:
        with open(args.inputs) as f:
            inputs = json.load(f)
    else:
        # Simple inline text input
        if not args.text:
            logger.error("Provide --inputs <file.json> or --text <sample text>")
            sys.exit(1)
        inputs = [{"id": "sample", "text": args.text}]

    checks = None
    if args.checks:
        checks_config = json.loads(args.checks)
        checks = build_checks_from_config(checks_config)

    lab = PromptLab(models=models)
    reports = lab.ab(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        inputs=inputs,
        prompt_a_id=args.a_id or "A",
        prompt_b_id=args.b_id or "B",
        checks=checks,
        llm_judge=args.llm_judge,
    )

    for model, report in reports.items():
        print(report.summary())


def cmd_regression_save(args):
    """Save a regression baseline."""
    from promptlab.lab import PromptLab
    from promptlab.scorers import RubricScorer

    config = load_test_file(args.test)
    models = args.models.split(",") if args.models else config.get("models", ["gpt-4o-mini"])
    checks_config = config.get("checks", [])
    checks = build_checks_from_config(checks_config) if checks_config else []

    prompt = list(config.get("prompts", {}).values())[0]
    inputs = config.get("inputs", [])

    lab = PromptLab(models=models)
    scorer = RubricScorer(checks=checks) if checks else RubricScorer()
    snapshot = lab.regression.save_baseline(
        name=args.name,
        prompt=prompt,
        inputs=inputs,
        models=models,
        scorer=scorer,
        overwrite=args.overwrite,
    )
    print(f"\n  Baseline '{args.name}' saved ({snapshot.created_at[:10]})")
    print(f"  Scores: {json.dumps(snapshot.scores, indent=2)}")


def cmd_regression_check(args):
    """Check a prompt against a saved baseline."""
    from promptlab.lab import PromptLab
    from promptlab.scorers import RubricScorer

    config = load_test_file(args.test)
    models = args.models.split(",") if args.models else config.get("models", ["gpt-4o-mini"])
    checks_config = config.get("checks", [])
    checks = build_checks_from_config(checks_config) if checks_config else []

    # Use updated prompt if provided, else first from config
    prompts = config.get("prompts", {})
    prompt = args.prompt or list(prompts.values())[0]
    inputs = config.get("inputs", [])

    lab = PromptLab(models=models)
    scorer = RubricScorer(checks=checks) if checks else RubricScorer()

    report = lab.regression.check(
        name=args.name,
        prompt=prompt,
        inputs=inputs,
        models=models,
        scorer=scorer,
    )
    print(report.summary())
    sys.exit(1 if report.has_regression else 0)


def cmd_list(args):
    """List saved baselines."""
    from promptlab.regression import RegressionTracker
    tracker = RegressionTracker(baselines_dir=args.baselines_dir or "baselines")
    baselines = tracker.list_baselines()
    if not baselines:
        print("No baselines saved yet.")
    else:
        print(f"\nSaved baselines ({len(baselines)}):")
        for name in sorted(baselines):
            print(f"  {name}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="promptlab",
        description="Prompt Testing Framework — evaluate, compare, and regression-test prompts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p_run = sub.add_parser("run", help="Run a batch evaluation")
    p_run.add_argument("test",          help="Path to test YAML/JSON file")
    p_run.add_argument("--models",      help="Comma-separated model IDs (overrides test file)")
    p_run.add_argument("--output",      help="Output directory (default: results/)")
    p_run.add_argument("--run-id",      help="Label for this run")
    p_run.add_argument("--llm-judge",   action="store_true")

    # ── ab ──
    p_ab = sub.add_parser("ab", help="A/B compare two prompts")
    p_ab.add_argument("--prompt-a",  required=True, help="Prompt A template")
    p_ab.add_argument("--prompt-b",  required=True, help="Prompt B template")
    p_ab.add_argument("--inputs",    help="JSON file with input list")
    p_ab.add_argument("--text",      help="Inline text for {{text}} variable")
    p_ab.add_argument("--models",    help="Comma-separated model IDs")
    p_ab.add_argument("--a-id",      help="Label for prompt A (default: A)")
    p_ab.add_argument("--b-id",      help="Label for prompt B (default: B)")
    p_ab.add_argument("--checks",    help="JSON array of check configs")
    p_ab.add_argument("--llm-judge", action="store_true")

    # ── regression ──
    p_reg = sub.add_parser("regression", help="Save or check regression baselines")
    reg_sub = p_reg.add_subparsers(dest="reg_command", required=True)

    p_save = reg_sub.add_parser("save", help="Save a baseline")
    p_save.add_argument("--name",      required=True)
    p_save.add_argument("--test",      required=True)
    p_save.add_argument("--models",    help="Override models")
    p_save.add_argument("--overwrite", action="store_true")

    p_check = reg_sub.add_parser("check", help="Check against a baseline")
    p_check.add_argument("--name",   required=True)
    p_check.add_argument("--test",   required=True)
    p_check.add_argument("--prompt", help="New prompt to test (overrides test file)")
    p_check.add_argument("--models", help="Override models")

    # ── list ──
    p_list = sub.add_parser("list", help="List saved baselines")
    p_list.add_argument("--baselines-dir", default="baselines")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "ab":
        cmd_ab(args)
    elif args.command == "regression":
        if args.reg_command == "save":
            cmd_regression_save(args)
        elif args.reg_command == "check":
            cmd_regression_check(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
