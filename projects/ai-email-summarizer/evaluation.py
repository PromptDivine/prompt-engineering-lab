"""
evaluation.py
=============
Email Summarizer — Evaluation Engine
Project: P6 · prompt-engineering-lab

Metrics per summary:
  rouge1 / rouge2 / rougeL  — lexical overlap with reference
  compression_ratio          — output length / input length
  word_count                 — output word count
  tone_preservation          — how well summary matches original tone
  latency_s                  — API response time
  flesch_kincaid             — readability of summary
"""

import re
import math
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from tone_detector import detect_tone, score_tone_preservation

logger = logging.getLogger(__name__)


@dataclass
class SummaryEval:
    email_id: str
    email_type: str       # single / thread
    email_tone: str       # original tone
    model: str
    prompt_id: str
    strategy: str
    summary: str
    latency_s: float
    # Metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    compression_ratio: float = 0.0
    word_count: int = 0
    tone_preservation: float = 0.0
    flesch_kincaid: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── ROUGE (pure Python) ──────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())

def _ngrams(tokens: list, n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts

def _rouge_n(hyp: str, ref: str, n: int) -> float:
    h, r = _tokenize(hyp), _tokenize(ref)
    hg, rg = _ngrams(h, n), _ngrams(r, n)
    overlap = sum(min(hg.get(g, 0), rg[g]) for g in rg)
    ref_total = sum(rg.values())
    hyp_total = sum(hg.values())
    recall    = overlap / ref_total if ref_total else 0.0
    precision = overlap / hyp_total if hyp_total else 0.0
    return round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

def _lcs(x: list, y: list) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if x[i-1] == y[j-1] else max(curr[j-1], prev[j])
        prev = curr
    return prev[n]

def _rouge_l(hyp: str, ref: str) -> float:
    h, r = _tokenize(hyp), _tokenize(ref)
    lcs = _lcs(h, r)
    recall    = lcs / len(r) if r else 0.0
    precision = lcs / len(h) if h else 0.0
    return round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

def compute_rouge(summary: str, reference: str) -> dict:
    return {
        "rouge1": _rouge_n(summary, reference, 1),
        "rouge2": _rouge_n(summary, reference, 2),
        "rougeL": _rouge_l(summary, reference),
    }


# ── Compression ratio ────────────────────────────────────────

def compute_compression(original: str, summary: str) -> float:
    orig_words = len(_tokenize(original))
    summ_words = len(_tokenize(summary))
    return round(summ_words / orig_words, 4) if orig_words else 0.0


# ── Flesch-Kincaid ───────────────────────────────────────────

def compute_fk_grade(text: str) -> float:
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words = _tokenize(text)
    if not sents or not words:
        return 0.0
    def syllables(w):
        c = len(re.findall(r'[aeiouy]+', w.lower()))
        if w.lower().endswith('e') and len(w) > 2:
            c -= 1
        return max(1, c)
    asl = len(words) / len(sents)
    asw = sum(syllables(w) for w in words) / len(words)
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


# ── Master evaluator ─────────────────────────────────────────

def evaluate_summary(
    email_id: str,
    email_body: str,
    email_tone: str,
    email_type: str,
    reference_summary: str,
    model: str,
    prompt_id: str,
    strategy: str,
    summary: str,
    latency_s: float,
) -> SummaryEval:

    result = SummaryEval(
        email_id=email_id,
        email_type=email_type,
        email_tone=email_tone,
        model=model,
        prompt_id=prompt_id,
        strategy=strategy,
        summary=summary,
        latency_s=latency_s,
    )

    rouge = compute_rouge(summary, reference_summary)
    result.rouge1 = rouge["rouge1"]
    result.rouge2 = rouge["rouge2"]
    result.rougeL = rouge["rougeL"]

    result.compression_ratio = compute_compression(email_body, summary)
    result.word_count         = len(_tokenize(summary))
    result.flesch_kincaid     = compute_fk_grade(summary)

    # Tone preservation
    original_tone_result = detect_tone(email_body)
    result.tone_preservation = score_tone_preservation(original_tone_result, summary)

    return result


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    email = "URGENT: Production server prod-us-east-1 went offline at 14:32 UTC. All customer-facing services unavailable. Sev-1. Backend team join incident bridge NOW. DevOps begin failover. Next update in 20 minutes."
    reference = "Production server down since 14:32 UTC (Sev-1). Backend to join incident bridge immediately, DevOps to initiate failover. Update in 20 minutes."
    summary   = "Sev-1 incident: prod-us-east-1 is down as of 14:32 UTC. Backend team must join the incident bridge now and DevOps should start failover. Next update in 20 min."

    result = evaluate_summary(
        email_id="E02", email_body=email, email_tone="urgent",
        email_type="single", reference_summary=reference,
        model="test", prompt_id="TL01", strategy="tldr",
        summary=summary, latency_s=1.2,
    )

    print(f"ROUGE-1: {result.rouge1}  ROUGE-L: {result.rougeL}")
    print(f"Compression: {result.compression_ratio}  Words: {result.word_count}")
    print(f"Tone preservation: {result.tone_preservation}")
    print(f"FK grade: {result.flesch_kincaid}")
