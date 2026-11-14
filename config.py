"""Central settings. Every value can be overridden with an environment variable."""
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
try:  # load GROQ_API_KEY / HF_TOKEN from .env when present
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"            # put the source PDFs here
WORK = DATA / "work"          # intermediate files (chunks, QA, caches)
OUT = Path(os.getenv("OUT_DIR", ROOT / "outputs"))
for _p in (RAW, WORK, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---- LLM APIs (LiteLLM model strings; needs GROQ_API_KEY) -----------------
# Groq free tier = 1000 requests/day + 8000 tokens/min PER MODEL, so work is spread across models.
# qwen3.8-27b also has a 1000 OUTPUT-tokens/min cap, so it only does short-output work (the gate).
GEN_MODEL = os.getenv("GEN_MODEL", "groq/openai/gpt-oss-120b")           # writes QA pairs
GATE_MODEL = os.getenv("GATE_MODEL", "groq/qwen/qwen3.8-27b")            # CuratorKIT hallucination gate (other family)
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "groq/openai/gpt-oss-20b")      # checks "unanswerable" questions
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-120b")       # final evaluation judge
# AuditKIT uses its own backend prefix syntax ("groq:<model>")
AUDIT_JUDGE = os.getenv("AUDIT_JUDGE", "groq:" + JUDGE_MODEL.split("groq/", 1)[-1])
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))                # free tier: keep low

# ---- Model being fine-tuned ----------------------------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "unsloth/Qwen3-4B-Instruct-2507")  # same weights, unsloth mirror
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", OUT / "adapter"))

# ---- Retrieval -------------------------------------------------------------
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
TOP_K = 4                    # passages shown to the model

# ---- Dataset sizes ---------------------------------------------------------
MAX_TRAIN_CHUNKS = int(os.getenv("MAX_TRAIN_CHUNKS", "200"))   # 200 fits the free tier; 350 if paid
MAX_TEST_CHUNKS = int(os.getenv("MAX_TEST_CHUNKS", "60"))
QUESTIONS_PER_CHUNK = 3
NEGATIVE_RATE = 0.35         # share of train QA also turned into "context lacks answer" examples
TEST_PER_CLASS = int(os.getenv("TEST_PER_CLASS", "90"))  # candidates per class before hand-check
SEED = 42
