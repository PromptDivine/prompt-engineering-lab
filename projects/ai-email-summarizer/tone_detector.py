"""
tone_detector.py
================
Email Summarizer — Tone Detector
Project: P6 · prompt-engineering-lab

Classifies email tone into 5 categories:
  formal    — business/professional language
  casual    — informal/conversational
  urgent    — time-sensitive / action-required
  negative  — complaint / frustration / conflict
  positive  — celebration / praise / good news

Also detects email type: single vs thread (multi-email chain)

Used to:
  1. Select the appropriate tone-matched prompt
  2. Score tone preservation in summaries
"""

import re
from dataclasses import dataclass


@dataclass
class ToneResult:
    primary_tone: str       # dominant tone label
    scores: dict            # { tone: confidence_score }
    is_thread: bool         # True if multi-email chain
    formality_score: float  # 0=very casual, 1=very formal
    urgency_score: float    # 0=not urgent, 1=very urgent
    sentiment_score: float  # -1=negative, 0=neutral, +1=positive


# ── Signal lexicons ──────────────────────────────────────────

FORMAL_SIGNALS = [
    r'\b(pursuant|herein|hereby|aforementioned|notwithstanding)\b',
    r'\b(please be advised|please find|kindly|sincerely|regards|dear)\b',
    r'\b(the undersigned|as per|in accordance|effective date)\b',
    r'\b(attached please find|further to|with reference to)\b',
    r'\b(i am writing to|this is to inform|please note that)\b',
]

CASUAL_SIGNALS = [
    r'\b(hey|hi there|sup|omg|lol|haha|btw|fyi|tbh)\b',
    r'[!]{2,}',                                          # multiple exclamation marks
    r'\b(gonna|wanna|gotta|kinda|sorta|stuff|things)\b',
    r'😀|😅|🎉|😂|👍|❤️|🙏|🤔|😊',                      # emoji
    r'\b(awesome|amazing|totally|literally|basically)\b',
    r"(it's|that's|they're|we're|i'm|you're|can't|won't|don't)",
]

URGENT_SIGNALS = [
    r'\b(urgent|asap|immediately|critical|emergency|sev-?1|severity)\b',
    r'\b(deadline|due by|required by|expires|before (end of|eod|cob))\b',
    r'\b(action required|response needed|reply needed|time.sensitive)\b',
    r'\b(all hands|do not delay|within \d+ (hours?|minutes?))\b',
    r'\b(NOW|URGENT|CRITICAL|ALERT|WARNING)\b',          # caps urgency
    r'\b(2 hours?|30 minutes?|today|this afternoon|by \d+:\d+)\b',
]

NEGATIVE_SIGNALS = [
    r'\b(disappointed|frustrated|unacceptable|appalling|disgusting)\b',
    r'\b(complaint|complain|problem|issue|failure|failed|broke|broken)\b',
    r'\b(demand|require|insist|escalate|escalating|legal action)\b',
    r'\b(still no|never|not once|again|yet another|nth time)\b',
    r'\b(refund|compensation|credit|chargeback|BBB|dispute)\b',
    r'\b(do better|this is unacceptable|i am writing to express)\b',
]

POSITIVE_SIGNALS = [
    r'\b(congratulations?|congrats|excellent|outstanding|thrilled)\b',
    r'\b(we won|signed|approved|promoted|success|celebrate|celebration)\b',
    r'\b(incredible|exceptional|proud|well deserved|fantastic)\b',
    r'\b(thank you|grateful|appreciate|welcome|excited)\b',
    r'🎉|🥂|🏆|🌟|⭐',
]

THREAD_SIGNALS = [
    r'^---\s*(original|reply|forward|re:|fwd:)',
    r'^from:\s+\w',
    r'^>+\s+',
    r're:\s+re:\s+',
    r'wrote:\s*\n',
]


def _count_signals(text: str, patterns: list) -> int:
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
    return count


def detect_tone(email_text: str) -> ToneResult:
    """
    Classify the tone of an email.

    Returns ToneResult with primary_tone, per-tone scores,
    thread detection, formality, urgency, and sentiment scores.
    """
    text = email_text.strip()
    lower = text.lower()

    # Thread detection
    is_thread = any(
        re.search(p, text, re.IGNORECASE | re.MULTILINE)
        for p in THREAD_SIGNALS
    )

    # Raw signal counts
    formal_count   = _count_signals(text, FORMAL_SIGNALS)
    casual_count   = _count_signals(text, CASUAL_SIGNALS)
    urgent_count   = _count_signals(text, URGENT_SIGNALS)
    negative_count = _count_signals(text, NEGATIVE_SIGNALS)
    positive_count = _count_signals(text, POSITIVE_SIGNALS)

    total = max(1, formal_count + casual_count + urgent_count + negative_count + positive_count)

    scores = {
        "formal":   round(formal_count   / total, 3),
        "casual":   round(casual_count   / total, 3),
        "urgent":   round(urgent_count   / total, 3),
        "negative": round(negative_count / total, 3),
        "positive": round(positive_count / total, 3),
    }

    # Primary tone = highest scoring, with urgency bias
    # (urgent emails should be flagged as urgent even if also formal)
    if urgent_count >= 3:
        primary = "urgent"
    else:
        primary = max(scores, key=scores.get)
        # Tie-break: if formal and casual are close, prefer formal
        if primary == "casual" and scores["formal"] >= scores["casual"] * 0.8:
            primary = "formal"

    # Continuous scores for evaluation
    word_count = max(1, len(text.split()))
    formality_score = min(1.0, formal_count / (word_count * 0.05))
    urgency_score   = min(1.0, urgent_count  / (word_count * 0.03))
    sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)

    return ToneResult(
        primary_tone=primary,
        scores=scores,
        is_thread=is_thread,
        formality_score=round(formality_score, 3),
        urgency_score=round(urgency_score, 3),
        sentiment_score=round(sentiment_score, 3),
    )


def select_tone_prompt_id(tone: str) -> str:
    """Map detected tone to the appropriate tone-matched prompt ID."""
    return {
        "formal":   "TM_FORMAL",
        "casual":   "TM_CASUAL",
        "urgent":   "TM_URGENT",
        "negative": "TM_NEGATIVE",
        "positive": "TM_POSITIVE",
    }.get(tone, "TM_FORMAL")


def score_tone_preservation(
    original_tone: ToneResult,
    summary: str,
) -> float:
    """
    Score how well the summary preserves the tone of the original email.
    Returns 0.0 to 1.0.
    """
    summary_tone = detect_tone(summary)

    # Primary tone match is most important
    primary_match = 1.0 if summary_tone.primary_tone == original_tone.primary_tone else 0.0

    # Continuous score similarity (cosine-ish)
    orig_vec = original_tone.scores
    summ_vec = summary_tone.scores

    dot  = sum(orig_vec[t] * summ_vec[t] for t in orig_vec)
    n_o  = sum(v**2 for v in orig_vec.values()) ** 0.5
    n_s  = sum(v**2 for v in summ_vec.values()) ** 0.5
    cosine = dot / (n_o * n_s) if n_o * n_s > 0 else 0.0

    # Weighted: primary match matters more
    return round(0.6 * primary_match + 0.4 * cosine, 4)


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("formal",   "Dear Sarah, I am writing to follow up on the Q3 budget review. Please be advised that three line items require your attention by Wednesday."),
        ("casual",   "Hey! Are you free Saturday? We're doing ramen on Morrison Street — the wait is like 2 hours but omg so worth it lol 😅"),
        ("urgent",   "URGENT: Production server down. All hands needed NOW. Join the incident bridge immediately. Response required within 30 minutes."),
        ("negative", "I am absolutely disgusted with your service. My order has not arrived. I demand a full refund immediately or I will file a BBB complaint."),
        ("positive", "Team — WE WON THE HENDERSON ACCOUNT!!! 🎉 3-year contract, $1.2M annually! Celebration drinks tonight!"),
    ]

    for expected, text in tests:
        result = detect_tone(text)
        match  = "✓" if result.primary_tone == expected else "✗"
        print(f"{match} Expected: {expected:10s}  Got: {result.primary_tone:10s}  Scores: {result.scores}")
