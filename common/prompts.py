"""Prompt formats shared by data building, training, evaluation and the demo.

The SAME system prompt + user format is used for training and inference, so
the fine-tuned model sees exactly the layout it learned.
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
