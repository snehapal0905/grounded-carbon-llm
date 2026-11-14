"""Gradio demo: base model vs fine-tuned model, side by side, same retrieved sources.
Cited sources are highlighted. Run on Kaggle/Colab GPU:  python app.py   (prints a public link)
"""
import csv
import html
import random
import re

import gradio as gr

from common.model import Generator
from common.prompts import REFUSAL, SYSTEM_STRICT, build_messages, cited_ids
from common.retrieval import Index, load_chunks
from config import OUT, TOP_K

chunks = load_chunks()
index = Index(chunks)
gen = Generator()

examples = []
try:
    with open(OUT / "test_candidates.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("keep", "1") != "0"]
    random.Random(1).shuffle(rows)
    examples = [r["question"] for r in rows if r["type"] == "answerable"][:5] + \
               [r["question"] for r in rows if r["type"] == "unanswerable"][:5]
except FileNotFoundError:
    pass


def render_answer(text):
    if text.strip().startswith(REFUSAL):
        return f"<div class='ans refusal'>🛑 {html.escape(text)}</div>"
    body = re.sub(r"\[S(\d+)\]", r"<span class='cite'>[S\1]</span>", html.escape(text))
    return f"<div class='ans'>{body}</div>"


def render_sources(ctx, cited):
    out = []
    for i, c in enumerate(ctx, 1):
        cls = "src cited" if i in cited else "src"
        out.append(f"<div class='{cls}'><b>[S{i}]</b> <i>{html.escape(c['doc'])}, p.{c['page']}</i>"
                   f"{' ✅ cited' if i in cited else ''}<br>{html.escape(c['text'])}</div>")
    return "".join(out)


def ask(question):
    if not question.strip():
        return "", "", ""
    ctx = index.search(question, k=TOP_K)
    msgs = build_messages(question, ctx, system=SYSTEM_STRICT)
    base = gen.generate([msgs], tuned=False)[0]
    tuned = gen.generate([msgs], tuned=True)[0]
    return render_answer(base), render_answer(tuned), render_sources(ctx, set(cited_ids(tuned)))


CSS = """
.ans{padding:12px;border-radius:8px;background:#f4f6f8;color:#111;font-size:15px;line-height:1.5}
.refusal{background:#fff4e5;border-left:4px solid #f59e0b}
.cite{background:#d1fae5;color:#065f46;padding:0 4px;border-radius:4px;font-weight:600}
.src{padding:10px;margin:6px 0;border:1px solid #e5e7eb;border-radius:8px;font-size:13px;color:#222;background:#fff}
.cited{border:2px solid #10b981;background:#ecfdf5}
"""

with gr.Blocks(css=CSS, title="Grounded Carbon-Market QA") as demo:
    gr.Markdown("# Grounded Carbon-Market QA\nThe same retrieved passages go to both models. "
                "The fine-tuned model answers only from them, cites them, or says **Not found**.")
    q = gr.Textbox(label="Question", placeholder="e.g. How long is the minimum crediting period under ...?")
    btn = gr.Button("Ask", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base model (strict prompt + RAG)")
            a_base = gr.HTML()
        with gr.Column():
            gr.Markdown("### Fine-tuned model (ours)")
            a_tuned = gr.HTML()
    gr.Markdown("### Retrieved sources (green = cited by the fine-tuned model)")
    srcs = gr.HTML()
    if examples:
        gr.Examples(examples, inputs=q)
    btn.click(ask, q, [a_base, a_tuned, srcs])
    q.submit(ask, q, [a_base, a_tuned, srcs])

if __name__ == "__main__":
    demo.launch(share=True)
