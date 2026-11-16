"""Prompt formats shared by data building, training, evaluation and the demo (v2: quote-then-answer).

v2 change (RAFT, Zhang et al. 2024; FRONT, Huang et al. 2024): when the answer is in the sources the
model first copies the exact supporting sentence, then answers with a citation:

    Quote: [S2] "<sentence copied verbatim from source 2>"
    Answer: <1-3 sentence answer> [S2]

When it is not in the sources it replies with exactly REFUSAL. The same system prompt is used for
training, for the strict baseline, and for the fine-tuned model, so gains come from training only.
"""
import re

REFUSAL = "Not found in the provided documents."

SYSTEM_STRICT = (
    "You are a compliance assistant for carbon-market rules. Answer ONLY from the "
    "numbered sources in the user's message.\n"
    "Rules:\n"
    "1. Every statement must be supported by the sources. Never use outside knowledge.\n"
    "2. Cite the sources you used with their tags, e.g. [S2] or [S1][S3].\n"
    f'3. If the sources do not contain the answer, reply exactly: "{REFUSAL}" Do not guess.\n'
    "4. Be concise: 1-4 sentences."
)

SYSTEM_QUOTE = (
    "You are a compliance assistant for carbon-market rules. Answer ONLY from the "
    "numbered sources in the user's message. Never use outside knowledge and never guess.\n"
    "If the sources contain the answer, reply in exactly this format:\n"
    'Quote: [S#] "<the supporting sentence, copied word for word from that source>"\n'
    "Answer: <1-3 sentence answer> [S#]\n"
    f"If the sources do not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}"
)

# Weak prompt for the "plain RAG" baseline.
SYSTEM_SIMPLE = "You are a helpful assistant. Use the context to answer the question."


def format_sources(chunks):
    return "\n\n".join(
        f"[S{i + 1}] ({c['doc']}, p.{c['page']})\n{c['text']}" for i, c in enumerate(chunks)
    )


def user_message(question, chunks):
    return f"Sources:\n{format_sources(chunks)}\n\nQuestion: {question}"


def build_messages(question, chunks, system=SYSTEM_STRICT, answer=None):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user_message(question, chunks)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


_REFUSAL_PATTERNS = re.compile(
    r"not found in the provided documents|(?:sources?|context|documents?|passages?) (?:do|does) not "
    r"(?:contain|provide|mention|specify|include|say)|no information (?:about|on|regarding)|"
    r"not (?:mentioned|specified|covered|stated) in the (?:sources?|context|documents?)|"
    r"cannot be (?:determined|answered) from",
    re.I,
)


def looks_like_refusal(text: str) -> bool:
    """Cheap heuristic (the LLM judge makes the final call during evaluation)."""
    return bool(_REFUSAL_PATTERNS.search(text or ""))


def cited_ids(text: str):
    return sorted({int(m) for m in re.findall(r"\[S(\d+)\]", text or "")})


# ---------------- v2 helpers: quote-then-answer ----------------
_SENT = re.compile(r"(?<=[.;:!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "is", "are", "be", "for", "on", "by", "with",
         "that", "this", "as", "at", "it", "from", "must", "shall", "s", "which", "under", "any", "its"}


def _content(t):
    return [w for w in _WORD.findall((t or "").lower()) if w not in _STOP]


def best_sentence(answer, text, max_words=60):
    """Sentence of `text` (verbatim) that best supports `answer` (content-word overlap)."""
    ans = set(_content(answer))
    best, best_s = "", -1.0
    for s in (x.strip() for x in _SENT.split(text or "")):
        if len(s.split()) < 4:
            continue
        toks = _content(s)
        if not toks:
            continue
        score = len(ans & set(toks)) + 0.01 * min(len(toks), 30)
        if score > best_s:
            best, best_s = s, score
    words = best.split()
    if len(words) > max_words:  # keep a verbatim window around the densest overlap
        hits = [i for i, w in enumerate(words) if set(_content(w)) & ans]
        mid = hits[len(hits) // 2] if hits else 0
        start = max(0, min(mid - max_words // 2, len(words) - max_words))
        best = " ".join(words[start:start + max_words])
    return best


def quote_answer(answer, quote, sid):
    answer = re.sub(r"\s*\[S\d+\]\s*$", "", answer.strip())
    return f'Quote: [S{sid}] "{quote}"\nAnswer: {answer} [S{sid}]'


def final_answer(text):
    """The user-facing answer: the part after 'Answer:' if the model used the quote format."""
    t = (text or "").strip()
    m = re.search(r"(?im)^\s*answer\s*:\s*(.*)\Z", t, re.S)
    return m.group(1).strip() if m else t


def quotes(text):
    """[(source_id, quoted_text)] from 'Quote: [S#] "..."' lines."""
    return [(int(a), b) for a, b in re.findall(r'\[S(\d+)\]\s*["“](.+?)["”]', text or "", re.S)
            if re.search(r"(?i)quote", text or "")]


def is_refusal(text):
    a = final_answer(text)
    return a.startswith(REFUSAL) or (len(a) < 200 and looks_like_refusal(a))


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())).strip()


def quote_is_verbatim(quote, source_text):
    q = _norm(quote)
    return bool(q) and q in _norm(source_text)
