"""
promptlab/scorers.py
====================
Scoring engines for prompt evaluation.

Three scorer types:
  RubricScorer    — rule-based, programmatic, zero API cost
  LLMJudgeScorer  — LLM-evaluated on custom dimensions
  CompositeScorer — weighted combination of multiple scorers

All scorers implement .score(output, input_vars, prompt) → dict
"""

import re
import json
import logging
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────

class BaseScorer:
    """All scorers must implement .score() returning a flat dict of metric→value."""

    def score(self, output: str, input_vars: dict = None, prompt: str = "") -> dict:
        raise NotImplementedError


# ──────────────────────────────────────────────
# RubricScorer — programmatic checks
# ──────────────────────────────────────────────

@dataclass
class RubricCheck:
    name: str
    fn: Callable[[str, dict], bool]
    weight: float = 1.0
    description: str = ""


class RubricScorer(BaseScorer):
    """
    Rule-based scorer using a list of programmatic checks.
    Each check is a function (output, input_vars) → bool.

    Usage:
        scorer = RubricScorer(checks=[
            RubricCheck("has_summary", lambda o, _: len(o) > 20),
            RubricCheck("no_preamble", lambda o, _: not o.lower().startswith("sure")),
            RubricCheck("under_100_words", lambda o, _: len(o.split()) <= 100),
        ])

    Built-in factory methods for common checks:
        RubricScorer.word_limit(100)
        RubricScorer.must_contain("conclusion")
        RubricScorer.must_not_contain("I cannot")
        RubricScorer.starts_with_pattern(r"^\d+\.")
    """

    def __init__(self, checks: list = None):
        self.checks = checks or []

    def add(self, name: str, fn: Callable, weight: float = 1.0, description: str = ""):
        self.checks.append(RubricCheck(name=name, fn=fn, weight=weight, description=description))
        return self

    def score(self, output: str, input_vars: dict = None, prompt: str = "") -> dict:
        input_vars = input_vars or {}
        results = {}
        weighted_scores = []

        for check in self.checks:
            try:
                passed = bool(check.fn(output, input_vars))
            except Exception as e:
                logger.warning(f"RubricCheck '{check.name}' error: {e}")
                passed = False

            results[f"rubric_{check.name}"] = 1.0 if passed else 0.0
            weighted_scores.append(check.weight if passed else 0.0)

        total_weight = sum(c.weight for c in self.checks)
        results["rubric_score"] = round(
            sum(weighted_scores) / total_weight, 4
        ) if total_weight > 0 else 0.0

        return results

    # ── Built-in check factories ─────────────────────────────

    @staticmethod
    def word_limit(max_words: int) -> "RubricCheck":
        return RubricCheck(
            f"word_limit_{max_words}",
            lambda o, _: len(o.split()) <= max_words,
            description=f"Output must be <= {max_words} words",
        )

    @staticmethod
    def word_minimum(min_words: int) -> "RubricCheck":
        return RubricCheck(
            f"word_min_{min_words}",
            lambda o, _: len(o.split()) >= min_words,
            description=f"Output must be >= {min_words} words",
        )

    @staticmethod
    def must_contain(phrase: str, case_sensitive: bool = False) -> "RubricCheck":
        def check(o, _):
            return (phrase in o) if case_sensitive else (phrase.lower() in o.lower())
        return RubricCheck(f"contains_{phrase[:20]}", check,
                           description=f"Must contain '{phrase}'")

    @staticmethod
    def must_not_contain(phrase: str, case_sensitive: bool = False) -> "RubricCheck":
        def check(o, _):
            return (phrase not in o) if case_sensitive else (phrase.lower() not in o.lower())
        return RubricCheck(f"not_contains_{phrase[:20]}", check,
                           description=f"Must NOT contain '{phrase}'")

    @staticmethod
    def starts_with_pattern(pattern: str) -> "RubricCheck":
        return RubricCheck(
            f"starts_with_pattern",
            lambda o, _: bool(re.match(pattern, o.strip())),
            description=f"Must start with pattern: {pattern}",
        )

    @staticmethod
    def contains_pattern(pattern: str, name: str = "pattern") -> "RubricCheck":
        return RubricCheck(
            name,
            lambda o, _: bool(re.search(pattern, o, re.IGNORECASE)),
            description=f"Must match regex: {pattern}",
        )

    @staticmethod
    def json_valid() -> "RubricCheck":
        def check(o, _):
            try:
                json.loads(o.strip().strip("`"))
                return True
            except Exception:
                return False
        return RubricCheck("json_valid", check, description="Output must be valid JSON")

    @staticmethod
    def numbered_list(min_items: int = 1) -> "RubricCheck":
        return RubricCheck(
            f"numbered_list_{min_items}",
            lambda o, _: len(re.findall(r'^\s*\d+[\.\)]\s+\S', o, re.MULTILINE)) >= min_items,
            description=f"Must have >= {min_items} numbered list items",
        )

    @staticmethod
    def no_refusal() -> "RubricCheck":
        REFUSALS = ["i cannot", "i'm unable", "i can't", "as an ai", "i am not able"]
        return RubricCheck(
            "no_refusal",
            lambda o, _: not any(r in o.lower() for r in REFUSALS),
            description="Must not contain refusal phrases",
        )


# ──────────────────────────────────────────────
# LLMJudgeScorer — LLM evaluates on dimensions
# ──────────────────────────────────────────────

DEFAULT_JUDGE_DIMENSIONS = [
    ("quality",    "Overall response quality (1=poor, 5=excellent)"),
    ("relevance",  "How relevant is the response to the prompt? (1=off-topic, 5=perfectly on-topic)"),
    ("accuracy",   "Factual accuracy (1=many errors, 5=fully accurate)"),
    ("conciseness","Is it appropriately concise? (1=too long/short, 5=just right)"),
]

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator of AI-generated text.

PROMPT GIVEN TO MODEL:
{prompt}

MODEL OUTPUT:
{output}

Score the output on each dimension (1-5):
{dimensions_str}

Respond ONLY with valid JSON, no other text:
{{
{json_fields}
  "overall": <average of all scores>,
  "rationale": "<one sentence>"
}}"""


class LLMJudgeScorer(BaseScorer):
    """
    Uses an LLM to score outputs on customizable dimensions.

    Usage:
        scorer = LLMJudgeScorer(
            client=PromptLabClient(),
            judge_model="gpt-4o-mini",
            dimensions=[
                ("faithfulness", "Does it accurately represent the source? (1-5)"),
                ("fluency",      "Is it well-written? (1-5)"),
            ]
        )
    """

    def __init__(
        self,
        client=None,
        judge_model: str = "gpt-4o-mini",
        dimensions: Optional[list] = None,
        judge_prompt_template: Optional[str] = None,
        reference_key: Optional[str] = None,
    ):
        self.client       = client
        self.judge_model  = judge_model
        self.dimensions   = dimensions or DEFAULT_JUDGE_DIMENSIONS
        self.template     = judge_prompt_template or DEFAULT_JUDGE_PROMPT
        self.reference_key = reference_key  # key in input_vars for reference answer

    def _get_client(self):
        if self.client:
            return self.client
        from .client import PromptLabClient
        return PromptLabClient()

    def score(self, output: str, input_vars: dict = None, prompt: str = "") -> dict:
        input_vars = input_vars or {}
        client = self._get_client()

        dims_str = "\n".join(f"- {name}: {desc}" for name, desc in self.dimensions)
        json_fields = "\n".join(f'  "{name}": <score>,' for name, _ in self.dimensions)

        filled_prompt = self.template.format(
            prompt=prompt[:2000],
            output=output[:2000],
            dimensions_str=dims_str,
            json_fields=json_fields,
        )

        result = client.call(model=self.judge_model, prompt=filled_prompt, temperature=0.0)

        if result.error:
            logger.warning(f"LLMJudge call failed: {result.error}")
            return {f"judge_{name}": 0.0 for name, _ in self.dimensions}

        try:
            raw = result.output.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            scores = {}
            for name, _ in self.dimensions:
                scores[f"judge_{name}"] = float(data.get(name, 0))
            scores["judge_overall"]   = float(data.get("overall", 0))
            scores["judge_rationale"] = str(data.get("rationale", ""))
            return scores
        except Exception as e:
            logger.warning(f"LLMJudge parse failed: {e} | raw: {result.output[:200]}")
            return {f"judge_{name}": 0.0 for name, _ in self.dimensions}


# ──────────────────────────────────────────────
# CompositeScorer — combines multiple scorers
# ──────────────────────────────────────────────

class CompositeScorer(BaseScorer):
    """
    Combines multiple scorers into one, merging all score dicts.
    Optionally computes a weighted composite score.

    Usage:
        scorer = CompositeScorer([
            (RubricScorer([...]),    weight=0.4),
            (LLMJudgeScorer(...),    weight=0.6),
        ])
        scores = scorer.score(output, input_vars, prompt)
        # scores["composite"] = weighted average
    """

    def __init__(self, scorers: list):
        """
        Args:
            scorers: list of (scorer, weight) tuples, or just scorer objects
        """
        self.scorers = []
        for item in scorers:
            if isinstance(item, tuple):
                scorer, weight = item
            else:
                scorer, weight = item, 1.0
            self.scorers.append((scorer, weight))

    def score(self, output: str, input_vars: dict = None, prompt: str = "") -> dict:
        all_scores = {}
        composite_parts = []

        for scorer, weight in self.scorers:
            s = scorer.score(output, input_vars, prompt)
            all_scores.update(s)

            # Use the primary score of each scorer for composite
            if isinstance(scorer, RubricScorer) and "rubric_score" in s:
                composite_parts.append(s["rubric_score"] * weight)
            elif isinstance(scorer, LLMJudgeScorer) and "judge_overall" in s:
                # Normalize 1-5 to 0-1
                composite_parts.append((s["judge_overall"] / 5.0) * weight)

        total_weight = sum(w for _, w in self.scorers)
        if composite_parts and total_weight > 0:
            all_scores["composite"] = round(sum(composite_parts) / total_weight, 4)

        return all_scores
