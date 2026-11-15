"""Thin LiteLLM wrapper: disk cache (re-runs cost nothing), rate-limit backoff, JSON parsing."""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from config import LLM_CONCURRENCY, WORK

_CACHE_PATH = WORK / "llm_cache.jsonl"
_cache = None
_lock = threading.Lock()


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.exists():
            with _CACHE_PATH.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        _cache[r["k"]] = r["v"]
                    except Exception:
                        pass
    return _cache


# Answers already produced by another model can be reused (e.g. Groq answers saved on the laptop
# before switching to a local Ollama model on Kaggle): CACHE_FALLBACK_MODELS="groq/openai/gpt-oss-120b,..."
_FALLBACKS = [m for m in os.getenv("CACHE_FALLBACK_MODELS", "").split(",") if m]
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))


def _key(model, messages, temperature, max_tokens):
    return hashlib.sha256(json.dumps([model, messages, temperature, max_tokens]).encode()).hexdigest()


def chat(model, messages, temperature=0.0, max_tokens=900, retries=8):
    import litellm
    litellm.suppress_debug_info = True
    key = _key(model, messages, temperature, max_tokens)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    for fb in _FALLBACKS:
        k2 = _key(fb, messages, temperature, max_tokens)
        if cache.get(k2):
            return cache[k2]
    delay = 4.0
    for attempt in range(retries):
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
            r = litellm.completion(model=model, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens,
                                   timeout=_TIMEOUT, **extra)  # never hang forever on a dead socket
            text = r.choices[0].message.content or ""
            break
        except Exception as e:  # rate limits / transient errors
            if attempt == retries - 1:
                raise
            msg = str(e)
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else delay
            if "rate" not in msg.lower() and "429" not in msg and "timeout" not in msg.lower():
                wait = delay
            time.sleep(min(wait, 120))
            delay = min(delay * 2, 60)
    with _lock:
        cache[key] = text
        with _CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": text}) + "\n")
    return text


def parse_json(text):
    """Pull the first JSON object/array out of a model reply."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


def pmap(fn, items, workers=LLM_CONCURRENCY, desc=""):
    """Parallel map with a progress bar; exceptions become None (logged)."""
    def safe(x):
        try:
            return fn(x)
        except Exception as e:
            print(f"  ! {desc} item failed: {str(e)[:200]}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(safe, items), total=len(items), desc=desc))
