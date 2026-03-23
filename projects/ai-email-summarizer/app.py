"""
app.py
======
Email Summarizer — Gradio Demo
Project: P6 · prompt-engineering-lab

Usage:
    pip install gradio>=4.0.0
    python app.py
    # Opens at http://127.0.0.1:7860
"""

import os
import time
import logging

logging.basicConfig(level=logging.WARNING)


# ── Sample emails ────────────────────────────────────────────

SAMPLE_EMAILS = {
    "🚨 Urgent: Server Down": (
        "Team,\n\n"
        "Our primary production server (prod-us-east-1) went offline at 14:32 UTC. "
        "All customer-facing services are currently unavailable. This is a Severity 1 incident.\n\n"
        "Current status: The on-call engineer is investigating. Initial assessment suggests "
        "a database connection pool exhaustion.\n\n"
        "Immediate actions needed:\n"
        "- Backend team: Join the incident bridge NOW at zoom.us/j/incident\n"
        "- DevOps: Begin failover procedure to prod-us-west-2\n"
        "- Communications: Draft customer notification within 15 minutes\n\n"
        "Next update in 20 minutes.\n\nTyler"
    ),
    "📧 Formal: Budget Review": (
        "Dear Sarah,\n\n"
        "I am writing to follow up on the Q3 budget review discussed during last Tuesday's "
        "leadership meeting. The finance committee identified three line items requiring "
        "immediate attention before the end-of-quarter close on Friday.\n\n"
        "First, the software licensing renewals ($42,000) need approval from your department "
        "by Wednesday COB. Second, contractor invoices totaling $18,500 remain pending your "
        "sign-off. Third, we have identified a $7,200 variance in the travel expense category "
        "requiring a written explanation for audit purposes.\n\n"
        "Could you confirm receipt and advise whether a call tomorrow morning (9-11 AM EST) "
        "would be convenient?\n\nBest regards,\nMarcus Chen\nFinance Director"
    ),
    "💬 Casual: Weekend Plans": (
        "Hey Jamie!\n\n"
        "Omg I've been meaning to text you but work has been absolutely insane this week — "
        "we finally shipped that feature and I basically haven't slept lol.\n\n"
        "Are you free this Saturday? A few of us are thinking of going to that new ramen place "
        "on Morrison Street. Apparently the wait is like 2 hours but totally worth it? "
        "Then maybe the rooftop bar if the weather holds up.\n\n"
        "Let me know! Also would it be weird to invite Priya? I feel like she's been a bit "
        "left out lately.\n\nAlex"
    ),
    "😤 Complaint: Bad Service": (
        "To whom it may concern,\n\n"
        "I am writing to express my extreme dissatisfaction with the service I received. "
        "I placed order #847291 three weeks ago and it has still not arrived. When I called "
        "your customer service line, I was put on hold for 47 minutes before being disconnected. "
        "I then sent two emails that went completely unanswered.\n\n"
        "This is completely unacceptable. I have been a loyal customer for six years. "
        "The item was a birthday gift for my mother and her birthday has now passed.\n\n"
        "I expect a full refund of $89.99 immediately. If I do not receive a response within "
        "24 hours, I will be filing a complaint with the Better Business Bureau.\n\nRobert Fitch"
    ),
    "🎉 Good News: Promotion": (
        "Dear Priya,\n\n"
        "I am absolutely thrilled to share that the leadership team has unanimously approved "
        "your promotion to Senior Software Engineer, effective December 1st!\n\n"
        "This reflects your exceptional contributions — particularly your leadership of the "
        "API migration project, your mentorship of three junior engineers, and your consistent "
        "delivery of high-quality work.\n\n"
        "Your new base salary will be $145,000 with a 2,000 RSU equity grant vesting over "
        "four years. A formal offer letter will follow from HR.\n\n"
        "We are so proud to have you on the team. This is very well deserved!\n\n"
        "With warm regards,\nDavid Park\nVP Engineering"
    ),
    "📨 Thread: Project Negotiation": (
        "--- Original Message (Nov 10) ---\n"
        "From: Claire Wong, Marketing Director\n\n"
        "Hi everyone, kicking off the website redesign project. Target launch: January 15th. "
        "Goals: modernize visual identity, improve mobile scores from 67 to 90+, reduce bounce "
        "rate 20%. Budget: $45,000. Claire\n\n"
        "--- Reply (Nov 12) ---\n"
        "From: Ben Okafor, Lead Developer\n\n"
        "Hi Claire, January 15 is aggressive — we need 10 weeks minimum (late January). "
        "Mobile performance target of 90+ may require complete image pipeline rebuild (3 weeks). "
        "Can we discuss? Ben\n\n"
        "--- Reply (Nov 14) ---\n"
        "From: Claire Wong\n\n"
        "Understood. CEO approved pushing to February 1st. Mobile target non-negotiable. "
        "Budget can stretch to $52,000 for a contractor. Claire\n\n"
        "--- Latest (Nov 15) ---\n"
        "From: Ben Okafor\n\n"
        "Perfect. With the extra budget I can bring in Maria Santos. "
        "Full project plan to you by end of week. Can design have mockups by Nov 25th? Ben"
    ),
}

# ── Prompt templates ─────────────────────────────────────────

STRATEGIES = {
    "TL;DR (1 sentence)": (
        "You are an executive assistant. Write a single-sentence TL;DR that gives a "
        "busy executive all they need to know.\n\nEmail:\n{email}\n\nTL;DR:"
    ),
    "Bullet Points": (
        "Summarize this email as 3-5 concise bullet points. "
        "Prioritize actions and decisions over background context.\n\n"
        "Email:\n{email}\n\nKey points:\n•"
    ),
    "Formal Paragraph": (
        "Write a professional 2-3 sentence executive summary of this email in third person, "
        "suitable for a business briefing document.\n\n"
        "Email:\n{email}\n\nExecutive summary:"
    ),
    "Casual Summary": (
        "Summarize this email in plain, casual language — like you're telling a friend "
        "what it said. 2-3 sentences max.\n\n"
        "Email:\n{email}\n\nQuick summary:"
    ),
    "Tone-Matched (Auto)": None,  # filled dynamically
}

TONE_PROMPTS = {
    "formal": (
        "Summarize the following formal business email. Maintain a professional and precise tone. "
        "Include key facts, required actions, and any deadlines.\n\n"
        "Email:\n{email}\n\nProfessional summary:"
    ),
    "casual": (
        "Summarize this casual/informal email in a friendly, conversational tone that matches "
        "the original style. Keep it relaxed and brief (2-3 sentences).\n\n"
        "Email:\n{email}\n\nCasual summary:"
    ),
    "urgent": (
        "This is an urgent email requiring immediate action. Summarize it with urgency intact. "
        "Lead with the critical action needed and any hard deadlines. Be direct.\n\n"
        "Email:\n{email}\n\nURGENT summary:"
    ),
    "negative": (
        "Summarize this complaint or negative feedback email factually and empathetically, "
        "capturing the sender's concerns accurately without minimizing them.\n\n"
        "Email:\n{email}\n\nSummary:"
    ),
    "positive": (
        "Summarize this positive or celebratory email with matching enthusiasm and warmth, "
        "preserving the good news and energy of the original.\n\n"
        "Email:\n{email}\n\nSummary:"
    ),
}

# ── Model config ─────────────────────────────────────────────

MODELS = {
    "GPT-4o-mini (OpenAI)":      ("openai",     "gpt-4o-mini"),
    "GPT-4o (OpenAI)":           ("openai",     "gpt-4o"),
    "Claude Haiku (Anthropic)":  ("anthropic",  "claude-haiku-4-5-20251001"),
    "Claude Sonnet (Anthropic)": ("anthropic",  "claude-sonnet-4-6"),
    "Mistral 7B (OpenRouter)":   ("openrouter", "mistralai/mistral-7b-instruct"),
    "Llama 3 8B (OpenRouter)":   ("openrouter", "meta-llama/llama-3-8b-instruct"),
}


# ── API call ─────────────────────────────────────────────────

def call_model(provider: str, model_id: str, prompt: str):
    t0 = time.time()
    try:
        if provider in ("openai", "openrouter"):
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY", "")
            kwargs = {"api_key": key}
            if provider == "openrouter":
                kwargs["base_url"] = "https://openrouter.ai/api/v1"
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip(), round(time.time() - t0, 2), None

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            resp = client.messages.create(
                model=model_id,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip(), round(time.time() - t0, 2), None

    except Exception as e:
        return None, round(time.time() - t0, 2), str(e)


# ── Core summarize function ───────────────────────────────────

def summarize(email_text, model_name, strategy_name):
    if not email_text or not email_text.strip():
        return (
            "⚠️ Please paste an email or load a sample first.",
            "",
            "",
            "",
        )

    # Detect tone
    try:
        from tone_detector import detect_tone
        tone_result = detect_tone(email_text)
        detected_tone = tone_result.primary_tone
        tone_info = (
            f"Detected tone: **{detected_tone.upper()}** | "
            f"Formality: {tone_result.formality_score:.2f} | "
            f"Urgency: {tone_result.urgency_score:.2f} | "
            f"Sentiment: {tone_result.sentiment_score:+.2f} | "
            f"{'Thread' if tone_result.is_thread else 'Single email'}"
        )
    except Exception as e:
        detected_tone = "formal"
        tone_info = f"Tone detection unavailable: {e}"

    # Build prompt
    if strategy_name == "Tone-Matched (Auto)":
        template = TONE_PROMPTS.get(detected_tone, TONE_PROMPTS["formal"])
        strategy_label = f"Tone-Matched → {detected_tone}"
    else:
        template = STRATEGIES.get(strategy_name, STRATEGIES["Bullet Points"])
        strategy_label = strategy_name

    prompt = template.format(email=email_text)

    # Call model
    if model_name not in MODELS:
        return "⚠️ Unknown model selected.", tone_info, "", ""

    provider, model_id = MODELS[model_name]
    output, latency, error = call_model(provider, model_id, prompt)

    if error or not output:
        api_key_hint = {
            "openai":     "OPENAI_API_KEY",
            "anthropic":  "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(provider, "API key")
        return (
            f"❌ Error: {error or 'Empty response'}\n\n"
            f"Make sure {api_key_hint} is set as an environment variable.",
            tone_info,
            "",
            f"Strategy used: {strategy_label}",
        )

    latency_text = f"⏱ {latency}s  |  {len(output.split())} words  |  {model_name}  |  {strategy_label}"
    prompt_preview = prompt[:600] + ("..." if len(prompt) > 600 else "")

    return output, tone_info, latency_text, prompt_preview


# ── Gradio UI ─────────────────────────────────────────────────

def build_app():
    import gradio as gr

    def load_sample(name):
        return SAMPLE_EMAILS.get(name, "")

    with gr.Blocks(title="AI Email Summarizer") as demo:

        gr.Markdown("# 📧 AI Email Summarizer\n**P6 · prompt-engineering-lab** — Tone-aware multi-strategy email summarization")

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="GPT-4o-mini (OpenAI)",
                    label="Model",
                )
                strategy_dd = gr.Dropdown(
                    choices=list(STRATEGIES.keys()),
                    value="Bullet Points",
                    label="Strategy",
                )
                gr.Markdown("---")
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLE_EMAILS.keys()),
                    label="Load a sample email",
                    value=None,
                )
                load_btn = gr.Button("Load Sample →", size="sm")

            with gr.Column(scale=2):
                email_input = gr.Textbox(
                    label="Email",
                    placeholder="Paste your email here, or load a sample on the left...",
                    lines=14,
                )

        summarize_btn = gr.Button("✨ Summarize", variant="primary")

        tone_out    = gr.Markdown(value="")
        summary_out = gr.Textbox(label="Summary", lines=6, interactive=False)
        latency_out = gr.Markdown(value="")

        with gr.Accordion("View prompt sent to model", open=False):
            prompt_out = gr.Textbox(label="Prompt", lines=8, interactive=False)

        gr.Markdown(
            "**Strategies:** TL;DR · Bullets · Formal · Casual · Tone-Matched (auto-detects tone)  \n"
            "**Models:** Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` env vars."
        )

        load_btn.click(fn=load_sample, inputs=sample_dd, outputs=email_input)
        sample_dd.change(fn=load_sample, inputs=sample_dd, outputs=email_input)
        summarize_btn.click(
            fn=summarize,
            inputs=[email_input, model_dd, strategy_dd],
            outputs=[summary_out, tone_out, latency_out, prompt_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
