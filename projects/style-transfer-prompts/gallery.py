"""
gallery.py
==========
Style Transfer Prompts — HTML Gallery Generator
"""

import argparse
from pathlib import Path
import html

import pandas as pd

RESULTS_DIR = Path("results")

STYLE_ORDER = ["journalism","academic","legal","executive","casual",
               "storytelling","technical","marketing","medical","minimalist"]

STYLE_EMOJI = {
    "journalism":   "📰",
    "academic":     "🎓",
    "legal":        "⚖️",
    "executive":    "💼",
    "casual":       "💬",
    "storytelling": "📖",
    "technical":    "🔧",
    "marketing":    "📣",
    "medical":      "🏥",
    "minimalist":   "✦",
}

STYLE_COLORS = {
    "journalism":   "#47c8ff",
    "academic":     "#b847ff",
    "legal":        "#ff8c47",
    "executive":    "#e8ff47",
    "casual":       "#47ffb2",
    "storytelling": "#ff47c8",
    "technical":    "#ff4776",
    "marketing":    "#ffd147",
    "medical":      "#47ffd4",
    "minimalist":   "#a0a8c8",
}


def generate_gallery(results_path=None, text_filter=None):
    results_path = results_path or RESULTS_DIR / "results.csv"
    df = pd.read_csv(results_path)

    # safe error filtering
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    else:
        df = df.copy()

    if text_filter:
        df = df[df["source_id"] == text_filter]

    source_texts = pd.read_csv(Path("data") / "source_texts.csv")
    source_map = dict(zip(source_texts["id"], source_texts["text"]))

    models = sorted(df["model"].unique())
    source_ids = sorted(df["source_id"].unique())

    data = {}
    for sid in source_ids:
        data[sid] = {}
        sub = df[df["source_id"] == sid]
        for style in STYLE_ORDER:
            style_sub = sub[sub["style"] == style]
            data[sid][style] = {}
            for model in models:
                msub = style_sub[style_sub["model"] == model]
                if not msub.empty:
                    data[sid][style][model] = msub.iloc[0].to_dict()

    html_out = _build_html(data, source_map, models, source_ids)

    RESULTS_DIR.mkdir(exist_ok=True)  # ensure dir exists
    out_path = RESULTS_DIR / "gallery.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Gallery saved: {out_path}")
    return out_path


def _metric_badges(row: dict) -> str:
    badges = []

    if pd.notna(row.get("fk_grade")):
        badges.append(f'<span class="badge">FK {row["fk_grade"]:.1f}</span>')

    if pd.notna(row.get("formality_score")):
        pct = int(row["formality_score"] * 100)
        badges.append(f'<span class="badge">Formality {pct}%</span>')

    if pd.notna(row.get("compression_ratio")):
        badges.append(f'<span class="badge">{row["compression_ratio"]:.2f}x len</span>')

    if pd.notna(row.get("latency_s")):
        badges.append(f'<span class="badge">{row["latency_s"]:.1f}s</span>')

    if pd.notna(row.get("judge_overall")):
        badges.append(f'<span class="badge judge">Judge {row["judge_overall"]:.1f}/5</span>')

    return " ".join(badges)


def _build_html(data, source_map, models, source_ids) -> str:
    all_sources_html = []

    # moved CSS here safely
    css_extra = """
    header { padding:40px 48px 24px; border-bottom:1px solid var(--border); }
    .header-tag { font-size:11px; color:var(--accent); letter-spacing:.2em; text-transform:uppercase; border:1px solid var(--accent); padding:4px 10px; display:inline-block; margin-bottom:16px; opacity:.8; }
    h1 { font-family:'Syne',sans-serif; font-size:42px; font-weight:800; letter-spacing:-.03em; line-height:.95; }
    h1 span { color:var(--accent); }
    .subtitle { font-size:13px; color:var(--muted); margin-top:12px; }

    .source-header { display:flex; align-items:center; gap:12px; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid var(--border); }
    .source-id { font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:var(--accent); }

    .split-view { display:grid; grid-template-columns:1fr 2fr; gap:2px; min-height:300px; }
    .source-pane, .output-pane { background:var(--surface); padding:24px; }
    .pane-label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.15em; margin-bottom:12px; }
    .source-text { font-size:13px; line-height:1.7; color:var(--text); opacity:.75; }

    .style-tabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }
    .style-tab {
        font-size:10px; padding:5px 12px; background:transparent;
        border:1px solid var(--border); color:var(--muted); cursor:pointer;
        font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em;
        transition:all .15s;
    }
    .style-tab:hover { border-color:var(--c,#47c8ff); color:var(--c,#47c8ff); }
    .style-tab.active { background:rgba(255,255,255,.05); border-color:var(--c,#47c8ff); color:var(--c,#47c8ff); }

    .model-tabs { display:flex; gap:4px; margin-bottom:12px; flex-wrap:wrap; }
    .model-tab {
        font-size:10px; padding:4px 10px; background:transparent;
        border:1px solid var(--border); color:var(--muted); cursor:pointer;
        font-family:'DM Mono',monospace; transition:all .15s;
    }
    .model-tab.active { border-color:var(--muted); color:var(--text); }
    .model-tab:hover { border-color:var(--text); }

    .badges { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
    .badge { font-size:10px; padding:3px 8px; background:var(--surface2); border:1px solid var(--border); color:var(--muted); }
    .badge.judge { border-color:var(--accent); color:var(--accent); }

    .output-text { font-size:13px; line-height:1.75; color:var(--text); white-space:pre-wrap; }

    .model-panel { display:none; }
    .model-panel.active { display:block; }
    """

    for sid in source_ids:
        source_text = html.escape(str(source_map.get(sid, "")))
        styles_html = []
        style_tabs = []

        styles_present = [s for s in STYLE_ORDER if s in data[sid] and data[sid][s]]

        for i, style in enumerate(styles_present):
            color = STYLE_COLORS.get(style, "#47c8ff")
            emoji = STYLE_EMOJI.get(style, "•")
            active = "active" if i == 0 else ""

            style_tabs.append(
                f'<button class="style-tab {active}" '
                f'data-style="{sid}-{style}" '
                f'style="--c:{color}" '
                f'onclick="switchStyle(this, \'{sid}\', \'{style}\')">'
                f'{emoji} {style}</button>'
            )

            model_panels = []
            for j, model in enumerate(models):
                row = data[sid][style].get(model, {})
                output_text = html.escape(str(row.get("output", "_No data for this model_")))
                badges = _metric_badges(row) if row else ""
                m_active = "active" if j == 0 else ""

                model_panels.append(f"""
                <div class="model-panel {m_active}" data-model="{model}" id="panel-{sid}-{style}-{model.replace(' ','-')}">
                  <div class="badges">{badges}</div>
                  <div class="output-text">{output_text}</div>
                </div>""")

            model_tabs_html = "".join(
                f'<button class="model-tab {"active" if j==0 else ""}" '
                f'data-model="{m}" '
                f'onclick="switchModel(this, \'{sid}\', \'{style}\')">{m}</button>'
                for j, m in enumerate(models)
            )

            display = "block" if i == 0 else "none"
            styles_html.append(f"""
            <div class="style-panel" id="style-{sid}-{style}" style="display:{display}">
              <div class="model-tabs">{model_tabs_html}</div>
              {"".join(model_panels)}
            </div>""")

        all_sources_html.append(f"""
        <div class="source-block" id="src-{sid}">
          <div class="source-header">
            <span class="source-id">{sid}</span>
            <span class="source-domain">{sid}</span>
          </div>
          <div class="split-view">
            <div class="source-pane">
              <div class="pane-label">SOURCE TEXT</div>
              <div class="source-text">{source_text}</div>
            </div>
            <div class="output-pane">
              <div class="style-tabs">{"".join(style_tabs)}</div>
              {"".join(styles_html)}
            </div>
          </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>P2 — Style Transfer Gallery</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
  :root {{
    --bg:#08090c; --surface:#0f1117; --surface2:#161820;
    --border:#1e2130; --text:#f0f2f8; --muted:#5a6080; --accent:#e8ff47;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Mono',monospace; }}
  {css_extra}
</style>
</head>
<body>
<main>
  {"".join(all_sources_html)}
</main>
<script>
  function switchStyle(btn, sid, style) {{
    // Deactivate all style tabs and panels for this source block
    const block = document.getElementById('src-' + sid);
    block.querySelectorAll('.style-tab').forEach(t => t.classList.remove('active'));
    block.querySelectorAll('.style-panel').forEach(p => p.style.display = 'none');

    // Activate the clicked tab and its panel
    btn.classList.add('active');
    const panel = document.getElementById('style-' + sid + '-' + style);
    if (panel) panel.style.display = 'block';
  }}

  function switchModel(btn, sid, style) {{
    // Deactivate all model tabs and panels within this style panel
    const stylePanel = document.getElementById('style-' + sid + '-' + style);
    stylePanel.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    stylePanel.querySelectorAll('.model-panel').forEach(p => p.classList.remove('active'));

    // Activate the clicked tab and its panel
    btn.classList.add('active');
    const model = btn.dataset.model;
    const modelPanel = stylePanel.querySelector('.model-panel[data-model="' + model + '"]');
    if (modelPanel) modelPanel.classList.add('active');
  }}
</script>
</body>
</html>"""


if __name__ == "__main__":
    generate_gallery()
