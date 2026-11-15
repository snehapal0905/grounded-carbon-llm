"""Step 6b - regenerate fine-tuned answers that came out empty.

The LoRA adapter sometimes emits a stray "<|im_start|>assistant" header, and a few answers stopped
immediately (empty). This re-runs only those questions with end-of-turn suppressed on the first token,
strips the stray header from every answer, and keeps the raw file as tuned_rag.raw.jsonl.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model import Generator, clean_output  # noqa: E402
from common.prompts import SYSTEM_STRICT, build_messages  # noqa: E402
from common.retrieval import load_chunks  # noqa: E402
from config import OUT  # noqa: E402


def main():
    p = OUT / "preds" / "tuned_rag.jsonl"
    raw = OUT / "preds" / "tuned_rag.raw.jsonl"
    if not raw.exists():
        shutil.copy(p, raw)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    for r in rows:
        r["output"] = clean_output(r["output"])
    todo = [r for r in rows if not r["output"]]
    print(f"{len(rows)} answers, {len(todo)} empty -> regenerating")
    if todo:
        by_id = {c["chunk_id"]: c for c in load_chunks()}
        gen = Generator()
        outs = gen.generate([build_messages(r["question"], [by_id[i] for i in r["ctx_ids"]],
                                            system=SYSTEM_STRICT) for r in todo], tuned=True)
        for r, o in zip(todo, outs):
            r["output"] = o
        print("still empty:", sum(not r["output"] for r in rows))
        for o in outs[:3]:
            print("-", o[:200])
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
