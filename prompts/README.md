# Prompt Strategy Library

> Cross-project index of prompt engineering strategies used across this portfolio.
> Each strategy links to real implementations in the project prompt libraries.

---

## Strategy Taxonomy

This portfolio uses 8 core prompt engineering strategies, each tested across multiple projects.
The table below maps strategy → project → measured impact.

| Strategy | Projects Used | Key Finding |
|----------|--------------|-------------|
| Zero-shot baseline | P1, P3, P7 | Control group — establishes baseline to measure improvement against |
| Few-shot | P1 | +8–15% ROUGE improvement over zero-shot on structured tasks |
| Role prompting | P1, P2, P6 | Strongest for tone-sensitive tasks; weaker on factual extraction |
| Chain-of-thought | P1, P2, P3, P7 | Biggest gains on reasoning (+23% vs zero-shot); marginal on summarization |
| Contrastive (DO/DON'T) | P2, P3 | Most reliable for negation constraints; reduces format violations by ~40% |
| Structured extraction | P2, P9 | Best precision on entity/date/value extraction tasks |
| Citation-enforced | P5, P8 | Reduces hallucination rate by largest margin of any single prompt change |
| Audience-targeted | P2, P6 | Most impactful for tone-matched output; drives highest tone preservation scores |

---

## Strategy Definitions

### 1. Zero-Shot Baseline
Minimal instruction with no examples or scaffolding. Used as the control group in every benchmark.

```
Summarize the following text:

{{text}}
```

📁 [P1 prompts](../projects/summarization-benchmark/prompts/prompts.txt) — `[P01]`

---

### 2. Role Prompting
Assigns an expert persona with a defined audience. Especially effective when the target output register differs from the default.

```
You are a senior reporter at a major national newspaper. Your editor has
asked you to rewrite the following text as a tight, factual news story.
```

📁 [P2 prompts](../projects/style-transfer-prompts/prompts/prompts.txt) — `[S01B]`, `[S02B]`, etc.
📁 [P6 prompts](../projects/email-summarizer/prompts/prompts.txt) — `[TL03]`, `[BL03]`, `[FP03]`

---

### 3. Chain-of-Thought (CoT)
Asks the model to reason step-by-step before producing a final answer. Consistently improves multi-step and reasoning tasks.

```
Work through this step by step before writing your final summary.

Step 1 — Identify the main topic:
Step 2 — List the 3 most important supporting points:
Step 3 — Write the final summary:
FINAL SUMMARY:
```

📁 [P1 prompts](../projects/summarization-benchmark/prompts/prompts.txt) — `[P07]`
📁 [P7 tasks](../projects/prompt-benchmark-system/tasks/task_definitions.py) — `"cot"` strategy

---

### 4. Contrastive Instruction (DO/DON'T)
Explicit positive and negative constraints. Most effective when the model has a tendency toward a specific unwanted behavior.

```
DO: Lead with the most important fact. Write in active voice.
DON'T: Use passive voice. Begin with background or context.
```

📁 [P2 prompts](../projects/style-transfer-prompts/prompts/prompts.txt) — `[S01C]`, `[S02C]`, etc.
📁 [P3 prompts](../projects/instruction-following/prompts/) — all `[NG*C]` negation tasks

---

### 5. Structured Extraction
Explicitly defines the output schema. Most reliable for information extraction tasks.

```
Provide your response in this exact format:

ONE-LINE SUMMARY: [Single sentence]
KEY FACTS:
1. [Most important fact]
2. [Second most important fact]
CONTEXT: [Why this matters]
```

📁 [P1 prompts](../projects/summarization-benchmark/prompts/prompts.txt) — `[P06]`
📁 [P9 intelligence](../projects/document-intelligence/intelligence.py) — `EXTRACT_PROMPT`

---

### 6. Citation-Enforced Grounding
Requires the model to quote the source text supporting every claim. The single most effective prompt change for reducing hallucination rate.

```
You MUST include a direct quote from the context that supports your answer.
ANSWER: [your answer]
QUOTE: "[exact quote from context]"
If the answer is not in the context: ANSWER: Not answerable / QUOTE: N/A
```

📁 [P5 prompts](../projects/grounded-qa/prompts/prompts.txt) — `[CT01]`, `[CT02]`, `[CT03]`
📁 [P8 mitigator](../projects/hallucination-detection/mitigator.py) — `CITATION_ENFORCED_PROMPT`

---

### 7. Audience-Targeted
Specifies a concrete target audience with characteristics. More specific than role prompting — defines who will read it, not just who wrote it.

```
Your audience is educated adults who want facts fast.
Aim for an 8th grade reading level.
```

📁 [P1 prompts](../projects/summarization-benchmark/prompts/prompts.txt) — `[P05]`
📁 [P2 prompts](../projects/style-transfer-prompts/prompts/prompts.txt) — `[P10]`

---

### 8. Self-Critique Loop
Ask the model to identify its own errors, then correct them. One of three mitigation strategies benchmarked in P8.

```
Step 1 — Identify any facts in the claim NOT supported by or contradicting the source:
[critique]

Step 2 — Write a corrected version fully faithful to the source:
CORRECTED CLAIM:
```

📁 [P8 mitigator](../projects/hallucination-detection/mitigator.py) — `SELF_CRITIQUE_PROMPT`

---

## Prompt Files by Project

| Project | Prompt File | Strategies |
|---------|------------|------------|
| P1 Summarization | [`summarization-benchmark/prompts/prompts.txt`](../projects/summarization-benchmark/prompts/prompts.txt) | zero_shot, instructed, role, structured, CoT, contrastive, compression, audience |
| P2 Style Transfer | [`style-transfer-prompts/prompts/prompts.txt`](../projects/style-transfer-prompts/prompts/prompts.txt) | 10 styles × 3 strategies (direct, role, contrastive) |
| P3 Instruction Following | [`instruction-following/`](../projects/instruction-following/) | rubric-based constraint tasks |
| P5 Grounded QA | [`grounded-qa/prompts/prompts.txt`](../projects/grounded-qa/prompts/prompts.txt) | ungrounded, grounded, cited |
| P6 Email Summarizer | [`email-summarizer/prompts/prompts.txt`](../projects/email-summarizer/prompts/prompts.txt) | TLDR, bullets, formal, casual, tone-matched |
| P7 Benchmark | [`prompt-benchmark-system/tasks/task_definitions.py`](../projects/prompt-benchmark-system/tasks/task_definitions.py) | zero_shot, instructed, CoT/role/test-driven per task |
| P8 Mitigation | [`hallucination-detection/mitigator.py`](../projects/hallucination-detection/mitigator.py) | grounded_rewrite, self_critique, citation_enforced |
| P9 Intelligence | [`document-intelligence/intelligence.py`](../projects/document-intelligence/intelligence.py) | classify, extract, QA |

---

## Key Takeaways

Based on measured results across all 9 projects:

1. **Chain-of-thought is most valuable on reasoning tasks** — marginal gains on pure summarization, large gains on multi-step problems.

2. **Citation enforcement is the single highest-ROI hallucination mitigation** — reduces hallucination rate more than any other single prompt change (P5, P8).

3. **Contrastive instructions outperform pure positive instructions for constraint compliance** — especially for negation tasks (P3).

4. **Role prompting excels at tone but not at factual precision** — good for style transfer (P2, P6), poor for grounded QA (P5).

5. **Structured output schemas improve extraction recall** — the explicit format reduces missed fields by 20–30% vs unstructured prompts (P9).

6. **Audience targeting + role prompting together outperform either alone** — tested in P1 `[P05]` vs `[P03]`.
