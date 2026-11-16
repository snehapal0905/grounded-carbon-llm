"""Step 4 (v2) - SFT set (quote-then-answer, completion-only), DPO preference pairs, test candidates.

Research basis (see reports/Grounded RAG finetuning methods.md):
  * RAFT (arXiv 2403.10131) / FRONT (2408.04568): gold passage among distractors, target first quotes
    the supporting sentence verbatim, then answers with a citation.
  * ChatQA (2401.10225): refusals help up to a point, then hurt -> refusal share capped at ~28%.
  * Trust-Align (2409.11242) + symmetric refusal-DPO (2603.19251): preference pairs where the SAME
    question appears with and without its evidence; every rejected answer carries a citation.

SFT rows are {"prompt": [system, user], "completion": [assistant]} so the loss covers only the answer
(the original repo trained on the ~2k-token prompt too).

Pair mix (N_PREF total, default 800):
  25.0%  answer_over_refusal       gold present: quote+answer  >  refusal          (fights over-refusal)
  50.0%  refusal_over_answer       evidence removed / unanswerable: refusal  >  cited made-up answer
  12.5%  right_over_wrong_citation gold present: correct [S#] + quote  >  wrong [S#] + other quote
  12.5%  faithful_over_contradiction gold present: correct  >  same answer with one fact changed

Outputs: outputs/sft/{train,validation}.jsonl, outputs/pref/train.jsonl, outputs/test_candidates.csv
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import chat, pmap  # noqa: E402
from common.prompts import (REFUSAL, SYSTEM_QUOTE, best_sentence, build_messages,  # noqa: E402
                            quote_answer)
from common.retrieval import Index, answer_recall, is_neighbor, load_chunks  # noqa: E402
from config import GEN_MODEL, OUT, SEED, TEST_PER_CLASS, TOP_K, WORK  # noqa: E402

NEG_RATE = float(os.getenv("NEG_RATE", "0.25"))          # share of positives also used as hard negatives
MAX_REFUSAL_SHARE = float(os.getenv("MAX_REFUSAL_SHARE", "0.28"))
N_PREF = int(os.getenv("N_PREF", "800"))

BAD_Q = re.compile(r"\b(the|this|above|given|provided) (passage|section|text|excerpt|paragraph)\b|"
                   r"according to the (passage|text)", re.I)

MADEUP_PROMPT = """A carbon-market analyst asks: "{q}"
Write the kind of confident, specific answer an overconfident assistant would give from memory:
1-2 sentences, include a concrete number, date, fee or rule. Output only the answer."""

CONTRADICT_PROMPT = """Rewrite this answer so it sounds equally confident but changes exactly ONE key fact
(a number, duration, percentage, named body, or requirement) to a wrong one. Output only the rewritten answer.

Answer: {a}"""


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def row(question, ctx, target):
    m = build_messages(question, ctx, system=SYSTEM_QUOTE)
    return {"prompt": m, "completion": [{"role": "assistant", "content": target}]}


def pair(question, ctx, chosen, rejected, kind):
    return {"prompt": build_messages(question, ctx, system=SYSTEM_QUOTE),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}], "kind": kind}


def main():
    rng = random.Random(SEED)
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    index = Index(chunks)
    train_mask = index.mask(splits={"train"})

    qa = read_jsonl(WORK / "qa_raw.jsonl")
    unans = read_jsonl(WORK / "unanswerable.jsonl")
    n0 = len(qa)
    qa = [r for r in qa if not BAD_Q.search(r["question"]) and len(r["answer"]) > 2]
    print(f"QA pairs: {n0} -> {len(qa)} after dropping non-self-contained questions")

    pos, negs, unas = [], [], []
    for r in qa:
        if r["split"] != "train":
            continue
        gold = by_id[r["chunk_id"]]
        cands = index.search(r["question"], k=TOP_K + 6, mask=train_mask)
        distract = [c for c in cands if c["chunk_id"] != gold["chunk_id"] and not is_neighbor(c, gold)]
        if len(distract) < TOP_K - 1:
            continue
        ctx = distract[:TOP_K - 1] + [gold]
        rng.shuffle(ctx)
        sid = ctx.index(gold) + 1
        quote = best_sentence(r["answer"], gold["text"])
        if not quote:
            continue
        pos.append({"q": r["question"], "a": r["answer"], "ctx": ctx, "sid": sid, "quote": quote,
                    "target": quote_answer(r["answer"], quote, sid)})
        if rng.random() < NEG_RATE:
            neg = [c for c in cands if c["chunk_id"] != gold["chunk_id"] and not is_neighbor(c, gold)
                   and answer_recall(r["answer"], c["text"]) < 0.5][:TOP_K]
            if len(neg) == TOP_K:
                negs.append({"q": r["question"], "a": r["answer"], "ctx": neg})
    for q in unans:
        if q["split"] == "train":
            unas.append({"q": q["question"], "kind": q["kind"],
                         "ctx": index.search(q["question"], k=TOP_K, mask=train_mask)})

    # ---- SFT set, refusal share capped (ChatQA: too many refusals hurts) ----
    refusals = [("hard_negative", n) for n in negs] + [("unanswerable", u) for u in unas]
    rng.shuffle(refusals)
    max_ref = int(MAX_REFUSAL_SHARE * len(pos) / (1 - MAX_REFUSAL_SHARE))
    refusals = refusals[:max_ref]
    sft = [("positive", row(p["q"], p["ctx"], p["target"])) for p in pos] + \
          [(k, row(x["q"], x["ctx"], REFUSAL)) for k, x in refusals]
    rng.shuffle(sft)
    n_val = max(int(len(sft) * 0.04), 16)
    d = OUT / "sft"
    d.mkdir(parents=True, exist_ok=True)
    for name, part in (("validation", sft[:n_val]), ("train", sft[n_val:])):
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for _, e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    sft_stats = Counter(k for k, _ in sft)
    print(f"SFT: {len(sft) - n_val} train / {n_val} val {dict(sft_stats)} "
          f"refusal share {1 - sft_stats['positive'] / len(sft):.0%}")

    # ---- DPO preference pairs ----
    n_ans, n_ref = int(N_PREF * 0.25), int(N_PREF * 0.5)
    n_cit, n_con = int(N_PREF * 0.125), int(N_PREF * 0.125)
    pairs = []
    P = pos[:]
    rng.shuffle(P)
    for p in P[:n_ans]:
        pairs.append(pair(p["q"], p["ctx"], p["target"], REFUSAL, "answer_over_refusal"))
    for p in P[n_ans:n_ans + n_cit]:
        wrong = rng.choice([i for i in range(1, TOP_K + 1) if i != p["sid"]])
        wq = best_sentence(p["a"], p["ctx"][wrong - 1]["text"])
        if wq:
            pairs.append(pair(p["q"], p["ctx"], p["target"], quote_answer(p["a"], wq, wrong),
                              "right_over_wrong_citation"))
    con_src = P[n_ans + n_cit:n_ans + n_cit + n_con]
    bad = pmap(lambda p: chat(GEN_MODEL, [{"role": "user", "content": CONTRADICT_PROMPT.format(a=p["a"])}],
                              temperature=0.7, max_tokens=300), con_src, desc="contradictions")
    for p, b in zip(con_src, bad):
        b = (b or "").strip()
        if b and b.lower() != p["a"].lower():
            pairs.append(pair(p["q"], p["ctx"], p["target"], quote_answer(b, p["quote"], p["sid"]),
                              "faithful_over_contradiction"))
    # refusal > cited answer: hard negatives (the real answer, but its evidence is gone) + unanswerables
    ref_src = [("hn", n) for n in negs] + [("un", u) for u in unas]
    rng.shuffle(ref_src)
    ref_src = ref_src[:n_ref]
    need = [x for k, x in ref_src if k == "un"]
    made = pmap(lambda u: chat(GEN_MODEL, [{"role": "user", "content": MADEUP_PROMPT.format(q=u["q"])}],
                               temperature=0.7, max_tokens=200), need, desc="made-up answers")
    made_by_q = {u["q"]: (m or "").strip() for u, m in zip(need, made)}
    for k, x in ref_src:
        ans = x["a"] if k == "hn" else made_by_q.get(x["q"], "")
        if not ans:
            continue
        sid = rng.randint(1, TOP_K)
        q = best_sentence(ans, x["ctx"][sid - 1]["text"])
        if q:
            pairs.append(pair(x["q"], x["ctx"], REFUSAL, quote_answer(ans, q, sid), "refusal_over_answer"))
    rng.shuffle(pairs)
    pd = OUT / "pref"
    pd.mkdir(parents=True, exist_ok=True)
    with open(pd / "train.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({k: p[k] for k in ("prompt", "chosen", "rejected")}, ensure_ascii=False) + "\n")
    pref_stats = Counter(p["kind"] for p in pairs)
    print(f"DPO pairs: {len(pairs)} {dict(pref_stats)}")

    # ---- test candidates (same construction as the original repo) ----
    test_qa = [r for r in qa if r["split"] == "test"]
    rng.shuffle(test_qa)
    per_chunk, picked = Counter(), []
    for r in test_qa:
        if per_chunk[r["chunk_id"]] < 2:
            per_chunk[r["chunk_id"]] += 1
            picked.append(r)
    picked = picked[:TEST_PER_CLASS]
    rows = [{"id": f"A{i:03d}", "type": "answerable", "kind": "", "question": r["question"],
             "gold_answer": r["answer"], "gold_chunk_id": r["chunk_id"],
             "source_text": by_id[r["chunk_id"]]["text"], "keep": 1} for i, r in enumerate(picked)]
    rows += [{"id": f"U{i:03d}", "type": "unanswerable", "kind": q["kind"], "question": q["question"],
              "gold_answer": REFUSAL, "gold_chunk_id": "", "source_text": "", "keep": 1}
             for i, q in enumerate([q for q in unans if q["split"] == "test"][:TEST_PER_CLASS])]
    with open(OUT / "test_candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Test candidates: {sum(r['type'] == 'answerable' for r in rows)} answerable, "
          f"{sum(r['type'] == 'unanswerable' for r in rows)} unanswerable")
    json.dump({"sft": dict(sft_stats), "sft_train": len(sft) - n_val, "pref": dict(pref_stats),
               "test_rows": len(rows)}, open(WORK / "sft_stats_v2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
