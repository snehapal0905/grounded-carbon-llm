"""Step 7 - score the 3 systems. Runs anywhere (needs GROQ_API_KEY, no GPU).

Two independent scorers:
  1. Grounding judge (LLM, per answer): refused? correct vs gold? # claims unsupported by sources?
     -> hallucination rate, correct-refusal rate, over-refusal rate, accuracy (with 95% bootstrap CIs)
  2. AuditKIT: claim-level `hallucination` (LLM judge) + `lexical_groundedness`, `context_coverage`,
     `f1_score`, and an AuditKIT base-vs-tuned comparison.

Writes outputs/results/{metrics.json, report.md, chart.png, judged_<system>.jsonl}
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import chat, parse_json, pmap  # noqa: E402
from common.prompts import cited_ids  # noqa: E402
from config import AUDIT_JUDGE, JUDGE_MODEL, OUT, TOP_K  # noqa: E402

SYSTEM_LABELS = {"base_rag": "Base + RAG", "base_strict": "Base + strict prompt + RAG",
                 "tuned_rag": "Fine-tuned + RAG (ours)"}
RES = OUT / "results"

JUDGE_PROMPT = """You grade a question-answering assistant that must answer ONLY from the given sources.

SOURCES:
{sources}

QUESTION: {question}
REFERENCE ANSWER: {gold}
ASSISTANT ANSWER: {output}

Decide:
- refused: true if the assistant says the information is not available / declines, WITHOUT giving a
  substantive answer. (An answer that corrects a false premise using the sources is NOT a refusal.)
- correct: compare with the reference. "yes" = same key facts, "partial" = some key facts or minor
  error, "no" = wrong or missing, "na" = no reference or the assistant refused.
- unsupported_claims: number of factual claims in the ASSISTANT ANSWER that the SOURCES do not
  support (outside knowledge counts as unsupported even if true). 0 for a refusal.
Return JSON only: {{"refused": true|false, "correct": "yes"|"partial"|"no"|"na", "unsupported_claims": <int>, "note": "<10 words>"}}"""


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def judge(p):
    gold = p["gold_answer"] if p["type"] == "answerable" else \
        "None - the sources are not expected to contain the answer; the ideal response is a refusal."
    j = parse_json(chat(JUDGE_MODEL, [{"role": "user", "content": JUDGE_PROMPT.format(
        sources="\n\n".join(p["ctx"]), question=p["question"], gold=gold, output=p["output"])}],
        max_tokens=1200))
    if not j:
        return None
    return {"refused": bool(j.get("refused")), "correct": str(j.get("correct", "na")).lower(),
            "unsupported_claims": int(j.get("unsupported_claims") or 0), "note": j.get("note", "")}


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ci(xs, n=1000, seed=0):
    if not xs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    boots = sorted(rate([rng.choice(xs) for _ in xs]) for _ in range(n))
    return [boots[int(0.025 * n)], boots[int(0.975 * n)]]


def metrics(rows):
    A = [r for r in rows if r["type"] == "answerable"]
    U = [r for r in rows if r["type"] == "unanswerable"]
    halluc = [r["j"]["unsupported_claims"] > 0 for r in rows]
    answered = [r for r in rows if not r["j"]["refused"]]
    m = {
        "n_answerable": len(A), "n_unanswerable": len(U),
        # headline: share of ALL responses containing >=1 claim not supported by the sources
        "hallucination_rate": rate(halluc), "hallucination_rate_ci95": ci(halluc),
        # on questions the docs don't answer: did it say so?
        "correct_refusal_rate": rate([r["j"]["refused"] for r in U]),
        "correct_refusal_ci95": ci([r["j"]["refused"] for r in U]),
        "unanswerable_hallucinated": rate([not r["j"]["refused"] and r["j"]["unsupported_claims"] > 0 for r in U]),
        # on questions the docs DO answer
        "accuracy": rate([r["j"]["correct"] == "yes" for r in A]),
        "accuracy_incl_partial": rate([r["j"]["correct"] in ("yes", "partial") for r in A]),
        "over_refusal_rate": rate([r["j"]["refused"] for r in A]),
        # over-refusal where the evidence WAS retrieved = the model's fault, not the retriever's
        "over_refusal_when_evidence_retrieved": rate([r["j"]["refused"] for r in A if r["retrieval_hit"]]),
        "citation_rate": rate([bool(cited_ids(r["output"])) for r in answered]),
        "retrieval_recall_at_k": rate([r["retrieval_hit"] for r in A]),
    }
    return m


def run_auditkit(preds_by_sys, use_llm_judge):
    """AuditKIT scoring on the precomputed outputs (no regeneration)."""
    import auditkit as ak
    scorers = ["lexical_groundedness", "context_coverage", "f1_score"]
    if use_llm_judge:
        from auditkit.metrics.rag_judge import Hallucination
        scorers = [Hallucination(judge_model=AUDIT_JUDGE, max_context_chars=2500)] + scorers
    runs, headlines = {}, {}
    for name, preds in preds_by_sys.items():
        samples = [ak.Sample(input=p["question"], target=p["gold_answer"], id=p["id"],
                             retrieval_context=p["ctx"], actual_output=p["output"],
                             metadata={"type": p["type"]}) for p in preds]
        res = ak.evaluate(samples, "precomputed", scorers, experiment_name=f"grounded-{name}")
        runs[name], headlines[name] = res, dict(res.headline)
        print(f"AuditKIT {name}: {res.headline}")
    if "base_strict" in runs and "tuned_rag" in runs:
        try:
            cmp = ak.compare(runs["base_strict"], runs["tuned_rag"])
            text = cmp.summary() if hasattr(cmp, "summary") else str(cmp)
            (RES / "auditkit_compare.txt").write_text(str(text), encoding="utf-8")
            print(text)
        except Exception as e:
            print(f"AuditKIT compare failed: {e}")
    return headlines


def chart(all_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("hallucination_rate", "Hallucination rate (lower better)"),
            ("correct_refusal_rate", "Correct refusal (higher better)"),
            ("accuracy_incl_partial", "Answer accuracy (higher better)"),
            ("over_refusal_rate", "Over-refusal (lower better)")]
    systems = list(all_m)
    colors = ["#9aa5b1", "#5b7083", "#1f9d55"]
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 3.8))
    for ax, (k, title) in zip(axes, keys):
        vals = [all_m[s][k] * 100 for s in systems]
        bars = ax.bar(range(len(systems)), vals, color=colors[:len(systems)])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([SYSTEM_LABELS[s].replace(" + ", "\n+ ") for s in systems], fontsize=7)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RES / "chart.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-auditkit", action="store_true")
    ap.add_argument("--no-auditkit-judge", action="store_true", help="AuditKIT lexical metrics only (saves API tokens)")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    preds_by_sys = {s: read_jsonl(OUT / "preds" / f"{s}.jsonl") for s in SYSTEM_LABELS
                    if (OUT / "preds" / f"{s}.jsonl").exists()}
    all_m = {}
    for name, preds in preds_by_sys.items():
        verdicts = pmap(judge, preds, desc=f"judge {name}")
        rows = [{**p, "j": v} for p, v in zip(preds, verdicts) if v is not None]
        if len(rows) < len(preds):
            print(f"  {len(preds) - len(rows)} judge failures skipped")
        with open(RES / f"judged_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_m[name] = metrics(rows)

    if not args.no_auditkit:
        try:
            for name, h in run_auditkit(preds_by_sys, not args.no_auditkit_judge).items():
                all_m[name]["auditkit"] = h
        except Exception as e:
            print(f"AuditKIT with LLM judge failed ({e}); retrying with AuditKIT's lexical metrics only.")
            try:
                for name, h in run_auditkit(preds_by_sys, False).items():
                    all_m[name]["auditkit"] = h
            except Exception as e2:
                print(f"AuditKIT step failed ({e2}); grounding-judge metrics are still saved.")

    json.dump(all_m, open(RES / "metrics.json", "w"), indent=2)
    pct = lambda x: "-" if x != x else f"{x * 100:.1f}%"  # noqa: E731
    lines = [f"| Metric | " + " | ".join(SYSTEM_LABELS[s] for s in all_m) + " |",
             "|---|" + "---|" * len(all_m)]
    for k, label in [("hallucination_rate", "Hallucination rate (any unsupported claim) ↓"),
                     ("correct_refusal_rate", "Correct refusal on unanswerable ↑"),
                     ("unanswerable_hallucinated", "Made-up answer on unanswerable ↓"),
                     ("accuracy", "Accuracy on answerable (strict) ↑"),
                     ("accuracy_incl_partial", "Accuracy on answerable (incl. partial) ↑"),
                     ("over_refusal_rate", "Over-refusal on answerable ↓"),
                     ("over_refusal_when_evidence_retrieved", "Over-refusal when evidence was retrieved ↓"),
                     ("citation_rate", "Answers with a source citation ↑")]:
        lines.append(f"| {label} | " + " | ".join(pct(all_m[s][k]) for s in all_m) + " |")
    any_m = next(iter(all_m.values()))
    lines.append(f"\nTest set: {any_m['n_answerable']} answerable + {any_m['n_unanswerable']} unanswerable "
                 f"questions (unanswerables verified against retrieval by an LLM; test chunks never seen in training). Retrieval recall@{TOP_K}: {pct(any_m['retrieval_recall_at_k'])}. "
                 f"Judge: {JUDGE_MODEL}.")
    for s, m in all_m.items():
        lo, hi = m["hallucination_rate_ci95"]
        lines.append(f"- {SYSTEM_LABELS[s]}: hallucination 95% CI [{pct(lo)}, {pct(hi)}]")
    report = "\n".join(lines)
    (RES / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    chart(all_m)
    print(f"\nSaved {RES}")


if __name__ == "__main__":
    main()
