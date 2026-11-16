"""Step 7 (v2) - score all systems with three independent checks + paired statistics.

  1. LLM judge from a DIFFERENT family than the Qwen student (default Gemma-3-12B on Ollama):
     refused? correct vs gold? number of claims unsupported by the sources?
  2. HHEM-2.1-open (Vectara, 0.1B NLI-style checker, no LLM): is the answer entailed by the sources?
  3. Deterministic checks (no judge): exact refusal string, citation correctness vs the gold passage,
     quotes copied verbatim from the cited source.
  + AuditKIT metrics and ak.compare(base_strict, tuned_dpo).

Headline metrics (Trust-Score style, Song et al. ICLR 2025): grounded-refusal F1 (balances refusing
when evidence is missing against answering when it is present), answer rate, over-refusal, accuracy,
hallucination among answered, citation precision/recall. Paired McNemar test + paired bootstrap CIs.

    python scripts/07_evaluate.py [--extra /path/to/other/preds:label ...] [--no-auditkit]
Writes outputs/results/{metrics.json, report.md, chart.png, judged_<system>.jsonl}
"""
import argparse
import os
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import chat, parse_json, pmap  # noqa: E402
from common.prompts import REFUSAL, cited_ids, final_answer, is_refusal, quote_is_verbatim, quotes  # noqa: E402
from config import AUDIT_JUDGE, JUDGE_MODEL, OUT, TOP_K  # noqa: E402

LABELS = {"base_rag": "Base + simple prompt", "base_strict": "Base + grounding prompt",
          "tuned_sft": "Ours: SFT", "tuned_dpo": "Ours: SFT + DPO",
          "main_tuned": "Main pipeline: fine-tuned"}
RES = Path(os.getenv("RESULTS_DIR", OUT / "results"))
PREDS = Path(os.getenv("PREDS_DIR", OUT / "preds"))

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
        sources="\n\n".join(p["ctx"]), question=p["question"], gold=gold, output=p["answer"])}],
        max_tokens=600))
    if not j:
        return None
    try:
        n = int(j.get("unsupported_claims") or 0)
    except (TypeError, ValueError):
        n = 0
    return {"refused": bool(j.get("refused")), "correct": str(j.get("correct", "na")).lower(),
            "unsupported_claims": n, "note": j.get("note", "")}


# ---------------- HHEM ----------------
_hhem = None


def hhem_scores(pairs):
    """P(consistent) for (premise, hypothesis) pairs; None if HHEM can't load."""
    global _hhem
    try:
        if _hhem is None:
            from transformers import AutoModelForSequenceClassification
            _hhem = AutoModelForSequenceClassification.from_pretrained(
                "vectara/hallucination_evaluation_model", trust_remote_code=True)
            try:
                _hhem = _hhem.to("cuda")
            except Exception:
                pass
            _hhem.eval()
        out = []
        for i in range(0, len(pairs), 8):
            s = _hhem.predict(pairs[i:i + 8])
            out += [float(x) for x in (s.tolist() if hasattr(s, "tolist") else s)]
        return out
    except Exception as e:
        print(f"HHEM unavailable ({str(e)[:200]}); skipping that check.")
        return None


# ---------------- stats ----------------
def rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, n=2000, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    b = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [b[int(0.025 * n)], b[int(0.975 * n)]]


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def mcnemar(a, b):
    """Exact two-sided McNemar p-value on paired booleans (a=baseline good, b=ours good)."""
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, n01, n10


def good(r):
    """Per-item 'did the right thing'."""
    if r["type"] == "unanswerable":
        return r["refused"]
    if r["refused"]:
        return not r.get("retrieval_hit", True)          # refusing is fine only if evidence wasn't retrieved
    return r["j"]["correct"] in ("yes", "partial") and r["j"]["unsupported_claims"] == 0


def enrich(p, v):
    r = {**p, "j": v}
    r.setdefault("answer", final_answer(p["output"]))
    # refusal = exact-string / heuristic OR judge says so
    r["refused"] = is_refusal(p["output"]) or v["refused"]
    r["exact_refusal"] = r["answer"].strip() == REFUSAL
    r["unsupported"] = (not r["refused"]) and v["unsupported_claims"] > 0
    return r


def metrics(rows):
    A = [r for r in rows if r["type"] == "answerable"]
    U = [r for r in rows if r["type"] == "unanswerable"]
    # Trust-Score style: an item "should be answered" iff it's answerable AND the evidence was retrieved
    should = [r for r in rows if r["type"] == "answerable" and r.get("retrieval_hit", True)]
    shouldnt = [r for r in rows if r not in should]
    answered = [r for r in rows if not r["refused"]]
    refused = [r for r in rows if r["refused"]]
    p_ans = rate([r in should for r in answered]) if answered else 0.0
    r_ans = rate([not r["refused"] for r in should]) if should else 0.0
    p_ref = rate([r in shouldnt for r in refused]) if refused else 0.0
    r_ref = rate([r["refused"] for r in shouldnt]) if shouldnt else 0.0
    cit = [r for r in answered if r in should and r.get("gold_pos")]
    cprec = [len(set(cited_ids(r["answer"])) & set(r["gold_pos"])) / len(cited_ids(r["answer"]))
             if cited_ids(r["answer"]) else 0.0 for r in cit]
    crec = [bool(set(cited_ids(r["answer"])) & set(r["gold_pos"])) for r in cit]
    qrows = [(r, q) for r in answered for q in quotes(r["output"])]
    qv = [quote_is_verbatim(q, r["ctx"][sid - 1] if 0 < sid <= len(r["ctx"]) else "") for r, (sid, q) in qrows]
    hall_all = [r["unsupported"] for r in rows]
    m = {
        "n": len(rows), "n_answerable": len(A), "n_unanswerable": len(U),
        "grounded_refusal_f1": (f1(p_ans, r_ans) + f1(p_ref, r_ref)) / 2,
        "answer_rate": len(answered) / len(rows),
        "correct_refusal_rate": rate([r["refused"] for r in U]),
        "correct_refusal_ci95": boot_ci([r["refused"] for r in U]),
        "exact_refusal_string": rate([r["exact_refusal"] for r in refused]),
        "over_refusal_rate": rate([r["refused"] for r in A]),
        "over_refusal_when_evidence_retrieved": rate([r["refused"] for r in should]),
        "accuracy": rate([r["j"]["correct"] == "yes" for r in A]),
        "accuracy_incl_partial": rate([r["j"]["correct"] in ("yes", "partial") for r in A]),
        "hallucination_rate": rate(hall_all), "hallucination_rate_ci95": boot_ci(hall_all),
        "hallucination_among_answered": rate([r["unsupported"] for r in answered]),
        "made_up_answer_on_unanswerable": rate([not r["refused"] for r in U]),
        "hhem_unsupported_among_answered": rate([r["hhem"] < 0.5 for r in answered if r.get("hhem") is not None]),
        "citation_rate": rate([bool(cited_ids(r["answer"])) for r in answered]),
        "citation_precision": rate(cprec), "citation_recall": rate(crec),
        "quote_verbatim_rate": rate(qv),
        "good_rate": rate([good(r) for r in rows]),
        "retrieval_recall_at_k": rate([r.get("retrieval_hit") for r in A]),
    }
    return m


def run_auditkit(rows_by_sys, use_llm_judge):
    import auditkit as ak
    scorers = ["lexical_groundedness", "context_coverage", "f1_score"]
    if use_llm_judge:
        from auditkit.metrics.rag_judge import Hallucination
        scorers = [Hallucination(judge_model=AUDIT_JUDGE, max_context_chars=2500)] + scorers
    runs, heads = {}, {}
    for name, rows in rows_by_sys.items():
        samples = [ak.Sample(input=r["question"], target=r["gold_answer"], id=r["id"],
                             retrieval_context=r["ctx"], actual_output=r["answer"],
                             metadata={"type": r["type"]}) for r in rows]
        res = ak.evaluate(samples, "precomputed", scorers, experiment_name=f"grounded-v2-{name}")
        runs[name], heads[name] = res, dict(res.headline)
        print(f"AuditKIT {name}: {res.headline}")
    if "base_strict" in runs and "tuned_dpo" in runs:
        try:
            cmp = ak.compare(runs["base_strict"], runs["tuned_dpo"])
            txt = cmp.summary() if hasattr(cmp, "summary") else str(cmp)
            (RES / "auditkit_compare.txt").write_text(str(txt), encoding="utf-8")
            print(txt)
        except Exception as e:
            print(f"AuditKIT compare failed: {e}")
    return heads


def chart(all_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("grounded_refusal_f1", "Grounded-refusal F1 ↑"),
            ("hallucination_among_answered", "Hallucination (answered) ↓"),
            ("correct_refusal_rate", "Correct refusal ↑"),
            ("over_refusal_when_evidence_retrieved", "Over-refusal ↓"),
            ("accuracy_incl_partial", "Accuracy ↑")]
    systems = list(all_m)
    colors = ["#9aa5b1", "#5b7083", "#7fb3d5", "#1f9d55", "#c0392b", "#8e44ad"]
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4))
    for ax, (k, title) in zip(axes, keys):
        vals = [(all_m[s][k] if all_m[s][k] == all_m[s][k] else 0) * 100 for s in systems]
        bars = ax.bar(range(len(systems)), vals, color=colors[:len(systems)])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([LABELS.get(s, s).replace(": ", ":\n") for s in systems], fontsize=7)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(RES / "chart.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="other preds files as path.jsonl:label")
    ap.add_argument("--no-auditkit", action="store_true")
    ap.add_argument("--no-auditkit-judge", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    preds = {s: read_jsonl(PREDS / f"{s}.jsonl") for s in LABELS if (PREDS / f"{s}.jsonl").exists()}
    for e in args.extra:
        path, label = e.rsplit(":", 1)
        preds[label] = read_jsonl(path)
        LABELS[label] = LABELS.get(label, label)
    rows_by_sys, all_m = {}, {}
    for name, ps in preds.items():
        for p in ps:
            p.setdefault("answer", final_answer(p["output"]))
        verdicts = pmap(judge, ps, desc=f"judge {name}")
        rows = [enrich(p, v) for p, v in zip(ps, verdicts) if v is not None]
        if len(rows) < len(ps):
            print(f"  {len(ps) - len(rows)} judge failures skipped")
        ans = [r for r in rows if not r["refused"]]
        hs = hhem_scores([("\n\n".join(r["ctx"])[:8000], re.sub(r"\[S\d+\]", "", r["answer"]).strip())
                          for r in ans]) if ans else []
        for r, h in zip(ans, hs or []):
            r["hhem"] = h
        with open(RES / f"judged_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows_by_sys[name], all_m[name] = rows, metrics(rows)

    # paired comparisons vs the strong baseline
    tests = {}
    if "base_strict" in rows_by_sys:
        base = {r["id"]: r for r in rows_by_sys["base_strict"]}
        for name, rows in rows_by_sys.items():
            if name == "base_strict":
                continue
            ids = [r["id"] for r in rows if r["id"] in base]
            mine = {r["id"]: r for r in rows}
            p, n01, n10 = mcnemar([good(base[i]) for i in ids], [good(mine[i]) for i in ids])
            diffs = [int(good(mine[i])) - int(good(base[i])) for i in ids]
            tests[name] = {"good_rate_gain": sum(diffs) / max(len(diffs), 1), "gain_ci95": boot_ci(diffs),
                           "mcnemar_p": p, "fixed": n01, "broke": n10, "n_paired": len(ids)}

    if not args.no_auditkit:
        try:
            for name, h in run_auditkit(rows_by_sys, not args.no_auditkit_judge).items():
                all_m[name]["auditkit"] = h
        except Exception as e:
            print(f"AuditKIT step failed ({e}); other metrics are still saved.")

    json.dump({"metrics": all_m, "paired_vs_base_strict": tests}, open(RES / "metrics.json", "w"), indent=2)
    pct = lambda x: "-" if x is None or x != x else f"{x * 100:.1f}"  # noqa: E731
    cols = list(all_m)
    lines = ["| Metric | " + " | ".join(LABELS.get(s, s) for s in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, label in [("grounded_refusal_f1", "Grounded-refusal F1 (Trust-Score style) ↑"),
                     ("good_rate", "Did the right thing (per item) ↑"),
                     ("hallucination_among_answered", "Hallucination among answered (judge) ↓"),
                     ("hhem_unsupported_among_answered", "Unsupported among answered (HHEM) ↓"),
                     ("hallucination_rate", "Responses with an unsupported claim ↓"),
                     ("correct_refusal_rate", "Correct refusal on unanswerable ↑"),
                     ("made_up_answer_on_unanswerable", "Answered an unanswerable question ↓"),
                     ("over_refusal_when_evidence_retrieved", "Over-refusal when evidence retrieved ↓"),
                     ("over_refusal_rate", "Over-refusal on all answerable ↓"),
                     ("accuracy", "Accuracy on answerable (strict) ↑"),
                     ("accuracy_incl_partial", "Accuracy on answerable (incl. partial) ↑"),
                     ("citation_precision", "Citation precision (cites the gold passage) ↑"),
                     ("citation_recall", "Citation recall ↑"),
                     ("quote_verbatim_rate", "Quotes copied verbatim ↑"),
                     ("exact_refusal_string", "Refusals using the exact sentence ↑"),
                     ("answer_rate", "Answer rate")]:
        lines.append(f"| {label} | " + " | ".join(pct(all_m[s].get(k)) for s in cols) + " |")
    m0 = next(iter(all_m.values()))
    lines.append(f"\nAll numbers are %. Test set: {m0['n_answerable']} answerable + {m0['n_unanswerable']} "
                 f"unanswerable. Retrieval recall@{TOP_K}: {pct(m0['retrieval_recall_at_k'])}%. "
                 f"Judge: {JUDGE_MODEL} (different family from the Qwen model being trained) + HHEM-2.1-open.")
    lines.append("\n**Paired vs Base + grounding prompt** (same questions, same passages):\n")
    for name, t in tests.items():
        lo, hi = t["gain_ci95"]
        lines.append(f"- {LABELS.get(name, name)}: right-thing rate {t['good_rate_gain'] * 100:+.1f} pts "
                     f"(95% CI {lo * 100:+.1f} to {hi * 100:+.1f}); fixed {t['fixed']}, broke {t['broke']}; "
                     f"McNemar p = {t['mcnemar_p']:.3g}")
    report = "\n".join(lines)
    (RES / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    chart(all_m)
    print(f"\nSaved {RES}")


if __name__ == "__main__":
    main()
