"""
evaluation.py
=============
Grounded QA — Evaluation Engine
Project: P5 · prompt-engineering-lab

Metrics:
  factual_accuracy     — does output match ground truth? (0/1 + partial)
  grounding_rate       — does answer stay within context? (no hallucinated facts)
  unanswerable_correct — did model correctly refuse unanswerable questions?
  citation_valid       — is the quoted passage actually in the context?
  citation_present     — did model include a citation at all?
  hallucination_flag   — binary: did any fabricated fact appear?

Hallucination taxonomy:
  NONE                 — fully grounded, no fabrications
  FABRICATED_FACT      — invented specific number/name/date not in context
  WRONG_SOURCE         — correct fact but attributed to wrong entity
  CONTEXT_LEAK         — answered unanswerable question with outside knowledge
  PARTIAL              — mostly correct but one unsupported claim
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class QAEvalResult:
    question_id: str
    condition: str          # ungrounded / grounded / cited
    prompt_id: str
    model: str
    question: str
    context: str
    ground_truth: str
    is_answerable: bool
    output: str
    latency_s: float

    # Scores
    factual_accuracy: float = 0.0       # 0, 0.5, or 1.0
    grounding_rate: float = 0.0         # 0 or 1.0 (no hallucinated facts)
    unanswerable_correct: float = 0.0   # 1.0 if model correctly refused
    citation_present: float = 0.0       # 1.0 if citation included
    citation_valid: float = 0.0         # 1.0 if quote exists in context
    hallucination_flag: int = 0         # 0=clean, 1=hallucinated
    hallucination_type: str = "NONE"

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Refusal detection
# ──────────────────────────────────────────────

REFUSAL_PATTERNS = [
    r"not (in|within|found in|present in|contained in) (the |this )?(context|passage|provided|text)",
    r"(context|passage|provided text) does not (contain|include|mention|state|specify)",
    r"(cannot|can't|unable to) (answer|find|determine).*(context|passage|provided)",
    r"not answerable from (the |this )?context",
    r"no information (about|on|regarding) .* (in|within) (the |this )?(context|passage)",
    r"the (provided )?(context|passage|information) (doesn't|does not)",
    r"not enough information",
    r"^not answerable",
    r"^n/a$",
]

def is_refusal(text: str) -> bool:
    """Returns True if the output is a proper refusal / 'not in context' response."""
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in REFUSAL_PATTERNS)


# ──────────────────────────────────────────────
# Factual accuracy
# ──────────────────────────────────────────────

def compute_factual_accuracy(output: str, ground_truth: str, is_answerable: bool) -> float:
    """
    Scores factual accuracy:
      1.0 = correct answer (or correct refusal for unanswerable)
      0.5 = partially correct (key fact present but incomplete/imprecise)
      0.0 = wrong or missing

    Uses lexical matching — for production use LLM judge here.
    """
    output_lower = output.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # For unanswerable questions: correct = model refused
    if not is_answerable:
        return 1.0 if is_refusal(output) else 0.0

    # For answerable questions: check if key facts from ground truth appear in output
    # Extract key tokens (numbers, proper nouns, quoted phrases)
    gt_tokens = _extract_key_facts(gt_lower)

    if not gt_tokens:
        # Fallback: substring match
        return 1.0 if gt_lower[:30] in output_lower else 0.0

    matches = sum(1 for t in gt_tokens if t in output_lower)
    match_rate = matches / len(gt_tokens)

    if match_rate >= 0.8:
        return 1.0
    elif match_rate >= 0.4:
        return 0.5
    else:
        return 0.0


def _extract_key_facts(text: str) -> list:
    """Extract numbers, percentages, proper nouns, and key phrases."""
    facts = []
    # Numbers and percentages
    facts.extend(re.findall(r'\b\d+\.?\d*\s*(?:percent|%|degrees|million|billion|thousand)?\b', text))
    # Quoted terms
    facts.extend(re.findall(r'"([^"]+)"', text))
    # All-caps words (acronyms) and capitalized phrases in original
    facts.extend(re.findall(r'\b[a-z]{3,}\b', text))
    return list(set(f.strip() for f in facts if len(f.strip()) > 2))


# ──────────────────────────────────────────────
# Grounding rate (hallucination detection)
# ──────────────────────────────────────────────

# Patterns that suggest the model is drawing on outside knowledge
OUTSIDE_KNOWLEDGE_SIGNALS = [
    r"(generally|typically|usually|often|commonly|in general)",
    r"(studies show|research suggests|experts say|scientists believe)",
    r"(for example|such as|like|including).{0,50}(which|that) (is|are|was|were)",
    r"(additionally|furthermore|moreover|also).{0,80}(not mentioned|beyond the)",
]

def compute_grounding_rate(output: str, context: str, is_answerable: bool) -> tuple:
    """
    Returns (grounding_score, hallucination_type).
    Checks whether output introduces facts not present in context.
    """
    output_lower = output.lower()
    context_lower = context.lower()

    # If model correctly refused on unanswerable → fully grounded
    if not is_answerable and is_refusal(output):
        return 1.0, "NONE"

    # If model answered an unanswerable question → context leak
    if not is_answerable and not is_refusal(output):
        return 0.0, "CONTEXT_LEAK"

    # Extract numbers from output and check against context
    output_numbers = set(re.findall(r'\b\d+\.?\d*\b', output_lower))
    context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context_lower))
    fabricated_numbers = output_numbers - context_numbers - {"1","2","3","4","5","0"}

    if fabricated_numbers:
        return 0.0, "FABRICATED_FACT"

    # Check for outside knowledge signals in answerable questions
    signals_found = sum(
        1 for p in OUTSIDE_KNOWLEDGE_SIGNALS
        if re.search(p, output_lower)
    )
    if signals_found >= 2:
        return 0.5, "PARTIAL"

    return 1.0, "NONE"


# ──────────────────────────────────────────────
# Citation validation
# ──────────────────────────────────────────────

def compute_citation_scores(output: str, context: str) -> tuple:
    """
    Returns (citation_present, citation_valid).

    citation_present: 1.0 if output contains a quoted passage
    citation_valid:   1.0 if that quoted passage exists in the context
    """
    # Find quoted text in output
    quotes = re.findall(r'"([^"]{10,})"', output)
    if not quotes:
        # Also check QUOTE: or EVIDENCE: prefixed lines
        lines = output.split("\n")
        for line in lines:
            if re.match(r'^(QUOTE|EVIDENCE|CITATION):\s*', line, re.IGNORECASE):
                rest = re.sub(r'^(QUOTE|EVIDENCE|CITATION):\s*"?', '', line, flags=re.IGNORECASE).rstrip('"')
                if len(rest) > 10:
                    quotes.append(rest)

    if not quotes:
        return 0.0, 0.0

    # Validate: check if each quote appears (approximately) in context
    context_lower = context.lower()
    valid_quotes = 0
    for quote in quotes:
        q_lower = quote.lower().strip()
        # Allow for minor whitespace/punctuation differences
        q_clean = re.sub(r'\s+', ' ', q_lower).strip()
        c_clean = re.sub(r'\s+', ' ', context_lower).strip()
        if q_clean in c_clean:
            valid_quotes += 1
        else:
            # Partial match — check if 70% of quote words appear in nearby context
            q_words = q_clean.split()
            if len(q_words) >= 3:
                matches = sum(1 for w in q_words if w in c_clean)
                if matches / len(q_words) >= 0.7:
                    valid_quotes += 0.5

    citation_present = 1.0
    citation_valid = min(1.0, valid_quotes / len(quotes)) if quotes else 0.0
    return citation_present, citation_valid


# ──────────────────────────────────────────────
# Master evaluation function
# ──────────────────────────────────────────────

def evaluate_qa(
    question_id: str,
    condition: str,
    prompt_id: str,
    model: str,
    question: str,
    context: str,
    ground_truth: str,
    is_answerable: bool,
    output: str,
    latency_s: float,
) -> QAEvalResult:

    result = QAEvalResult(
        question_id=question_id,
        condition=condition,
        prompt_id=prompt_id,
        model=model,
        question=question,
        context=context,
        ground_truth=ground_truth,
        is_answerable=is_answerable,
        output=output,
        latency_s=latency_s,
    )

    result.factual_accuracy = compute_factual_accuracy(output, ground_truth, is_answerable)

    grounding, hall_type = compute_grounding_rate(output, context, is_answerable)
    result.grounding_rate     = grounding
    result.hallucination_type = hall_type
    result.hallucination_flag = 0 if hall_type == "NONE" else 1

    # Unanswerable correctness (only meaningful for unanswerable questions)
    if not is_answerable:
        result.unanswerable_correct = 1.0 if is_refusal(output) else 0.0

    # Citation scores (only for cited condition)
    if condition == "cited":
        cit_present, cit_valid = compute_citation_scores(output, context)
        result.citation_present = cit_present
        result.citation_valid   = cit_valid

    return result


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ctx = "The new cells achieved 24.1 percent efficiency. Stanford researchers led the project."
    gt  = "24.1 percent efficiency"

    # Test 1: correct grounded answer
    r1 = evaluate_qa("Q01","grounded","GR01","test",
                     "What efficiency?", ctx, gt, True,
                     "The cells achieved 24.1 percent efficiency.", 1.0)
    print(f"Correct answer  → accuracy={r1.factual_accuracy}  grounding={r1.grounding_rate}  hall={r1.hallucination_type}")

    # Test 2: hallucinated answer
    r2 = evaluate_qa("Q01","grounded","GR01","test",
                     "What efficiency?", ctx, gt, True,
                     "The cells achieved 27.5 percent efficiency.", 1.0)
    print(f"Hallucinated    → accuracy={r2.factual_accuracy}  grounding={r2.grounding_rate}  hall={r2.hallucination_type}")

    # Test 3: correct refusal
    r3 = evaluate_qa("Q03","grounded","GR01","test",
                     "What is the rollback procedure?", ctx,
                     "The context does not describe rollback.", False,
                     "The provided context does not contain enough information to answer this question.", 1.0)
    print(f"Correct refusal → accuracy={r3.factual_accuracy}  unanswerable={r3.unanswerable_correct}  hall={r3.hallucination_type}")

    # Test 4: failed refusal (context leak)
    r4 = evaluate_qa("Q03","grounded","GR01","test",
                     "What is the rollback procedure?", ctx,
                     "The context does not describe rollback.", False,
                     "You should revert to the previous version by restoring from backup.", 1.0)
    print(f"Context leak    → accuracy={r4.factual_accuracy}  unanswerable={r4.unanswerable_correct}  hall={r4.hallucination_type}")
