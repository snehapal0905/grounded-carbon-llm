# Grounded Carbon LLM: answers only from the documents, says "Not found" otherwise

> A small open model (**Qwen3-4B-Instruct-2507**) fine-tuned to answer carbon-market compliance
> questions **only** from official standards documents. It **cites** the passage it used and replies
> **"Not found in the provided documents."** when the answer is not there.
> It is compared head-to-head against the same model with RAG and with a strict grounding prompt.

**Hackathon track:** Track 1, Fine-tuning & Data Curation (Lexsi Labs toolkits: **CuratorKIT → AlignTune → AuditKIT**)
**Problem statement:** *Grounded answers and fewer hallucinations.* Build a training set from an
organisation's own documents, train a model that answers only from them, says so when the answer is
missing, and makes measurably fewer unsupported claims.

> 🚧 **Status: running.** The full pipeline is running end to end on a free Colab T4 GPU
> ([`kaggle/colab_run.ipynb`](kaggle/colab_run.ipynb)). The results table below will be filled in from
> `outputs/results/report.md`. See [Progress](#progress).

## The three Lexsi Labs toolkits: where each one is used
| Stage | Toolkit | Where in the code | What it does here |
|---|---|---|---|
| **Build the data** | **CuratorKIT** | [`scripts/02_curate.py`](scripts/02_curate.py), `run_curatorkit()` | `Curator(CuratorConfig(...)).run()`: schema gate → exact + **MinHash dedup** → text cleaning → **QA generation** (3 per chunk) → **HallucinationGate** (judge model, threshold 0.7) → Alpaca export. Its `stage_counts` give the data funnel. |
| **Train** | **AlignTune** | [`scripts/05_train.py`](scripts/05_train.py) | `aligntune.create_sft_trainer(...)`: SFT with **LoRA** (r=16, α=32) on Qwen3-4B-Instruct, 4-bit, Unsloth backend, chat-template `messages` data. |
| **Evaluate** | **AuditKIT** | [`scripts/07_evaluate.py`](scripts/07_evaluate.py) | `ak.evaluate(...)` with `Hallucination` (LLM judge), `lexical_groundedness`, `context_coverage`, `f1_score` on base vs fine-tuned answers, then `ak.compare(...)`. |

---

## Table of contents
1. [Why this problem](#why-this-problem)
2. [What the model does](#what-the-model-does)
3. [The document corpus](#the-document-corpus)
4. [Pipeline overview](#pipeline-overview)
5. [Step-by-step: what each script does](#step-by-step-what-each-script-does)
6. [Key design decisions](#key-design-decisions)
7. [Evaluation: how "fewer hallucinations" is measured](#evaluation-how-fewer-hallucinations-is-measured)
8. [Tech stack](#tech-stack)
9. [Repository layout](#repository-layout)
10. [How to run](#how-to-run)
11. [Progress](#progress)
12. [Limitations](#limitations)

---

## Why this problem
Carbon credits are worth money only if they follow the rules of a standard: Verra VCS, Gold Standard,
ICVCM Core Carbon Principles, UNFCCC Article 6.4, or India's Carbon Credit Trading Scheme. Analysts
ask questions like *"How long is the crediting period?"* or *"What happens if a reversal exceeds the
buffer contribution?"*

General LLMs answer these **confidently and often wrongly**. They mix up standards, invent percentages,
or accept a false premise. In compliance work, a confident wrong answer is worse than no answer.
This project trains a model that:

- answers **only** from the organisation's documents,
- **cites** which retrieved passage supports each answer (`[S1]`, `[S2]` …),
- **refuses** with an exact, machine-checkable sentence when the documents don't contain the answer,
- and does this **without becoming over-cautious** (over-refusal is tracked as a first-class metric).

## What the model does
```
Question:  Under the ICVCM framework, what must a Jurisdictional REDD+ proponent do if a
           reversal exceeds its contributions to the pooled buffer reserve?
Sources:   [S1] … [S2] … [S3] … [S4] …   (top-4 passages from hybrid retrieval)

Answer:    The proponent must replenish the pooled buffer reserve so that it returns to a
           percentage proportionate to the reversal risk … [S2]

Question:  What is the CORSIA eligibility deadline for Gold Standard credits issued in 2025?
Answer:    Not found in the provided documents.
```

## The document corpus
Seven public, official documents (≈20 MB of PDFs → **445 text chunks** of ~300 words):

| Document | Publisher | Chunks (train / test / buffer) |
|---|---|---|
| VCS Standard v5.0 | Verra | 105 / 6 / 2 |
| VCS Program Guide v5.0 | Verra | 26 / 0 / 0 |
| Principles & Requirements v2.1 | Gold Standard | 39 / 6 / 2 |
| Core Carbon Principles v3 (framework + procedure) | ICVCM | 120 / 30 / 8 |
| Article 6.4 Activity Standard for Projects v2.0 | UNFCCC | 44 / 18 / 6 |
| Article 6.4 Methodology Standard (A6.4-SBM014-A05) | UNFCCC | 16 / 6 / 1 |
| Carbon Credit Trading Scheme 2023 (Gazette S.O. 2825(E)) | Govt. of India, Ministry of Power | 10 / 0 / 0 |
| **Total** | | **360 / 66 / 19** |

Exact versions and download links: [`docs/SOURCES.md`](docs/SOURCES.md). The PDFs are not committed;
download them into `data/raw/`.

## Pipeline overview
```
 PDFs (7 standards)
   │
   ▼ 01_ingest ──────────► 445 chunks, block-level train/test split + buffer chunks (no leakage)
   │
   ├─▶ 02_curate (CuratorKIT) ──► QA generation (gpt-oss-120b)
   │                              → hallucination gate (Qwen3 27B, different model family)
   │                              → MinHash dedup → clean grounded QA pairs
   │
   ├─▶ 03_unanswerable ─────────► "not in the docs" questions, 3 kinds
   │                              (missing detail / out of scope / false premise),
   │                              each VERIFIED against real retrieval
   │
   ▼ 04_build_sets ─────────────► SFT set (chat "messages" format):
   │                                • positives: question + 4 sources → answer with [S#]
   │                                • hard negatives: SAME question, evidence removed → refusal
   │                                • unanswerables → refusal
   │                              + test_candidates.csv (hand-checked by a human)
   │
   ▼ 05_train (AlignTune) ──────► LoRA SFT on Qwen3-4B (Unsloth, 4-bit, T4 GPU on Kaggle)
   │
   ▼ 06_generate ───────────────► 3 systems answer the same test questions with the same passages
   │
   ▼ 07_evaluate (AuditKIT) ────► hallucination rate, correct refusal, over-refusal, accuracy,
                                  citation rate, retrieval recall + AuditKIT metrics → report + chart
```

## Step-by-step: what each script does

### `scripts/01_ingest.py`: PDFs → chunks
- Extracts text with `pypdf` and splits it into **300-word chunks with 50-word overlap**.
- **Quality filter:** drops chunks that are mostly numbers/tables (alphabetic share < 0.55) or non-English
  (ASCII share < 0.9). This removes the Hindi half of the Indian gazette.
- **Block-level split:** groups 6 consecutive chunks into a block and sends ~12% of blocks to *test*.
  Chunks next to a test block become **buffer** and are used by nobody. Overlapping chunks would
  otherwise leak test text into training.
- Output: `data/work/chunks.jsonl`, `data/work/ingest_stats.json`.

### `scripts/02_curate.py`: CuratorKIT: chunks → grounded QA pairs
- Samples up to 200 train chunks and 60 test chunks. Each is prefixed with `Document: <name>` so
  questions name the standard ("Under the VCS Standard, …") instead of "according to the passage".
- Runs a **CuratorKIT** pipeline:
  `JSONLReader → SchemaGate → ExactDeduplicator → MinHashDeduplicator → TextCleaner → QAGenerationTask (3 Q/chunk) → HallucinationGate (threshold 0.7) → AlpacaExporter`
- The **generator** (`openai/gpt-oss-120b`) and the **hallucination judge** (`qwen/qwen3.8-27b`) are
  different model families, so the judge doesn't share the generator's blind spots.
- Maps every QA pair back to its source chunk (normalised text match, falling back to embedding
  nearest-neighbour) so it inherits the chunk's train/test split.
- Saves the **data funnel** (how many samples survived each stage) to `data/work/funnel.json`.
- `--engine fallback` is an emergency path with plain LLM calls and the same stages. **The submitted run uses CuratorKIT.**

Sample pairs from the CuratorKIT smoke run (12 chunks → 36 pairs, 36/36 passed the gate):

| Question | Answer (truncated) |
|---|---|
| Which elements are assessed during a Design Certification Renewal according to the Gold Standard requirements? | (a) changes in the project related to the General Eligibility Criteria; (b) incorporation of relevant updates to the Gold Standard Requirements; (c) re-definition of the baseline scenario … |
| What are the three specific requirements a carbon-crediting program must satisfy before it can grant the CCP Attribute of quantified positive SDG impacts? | (1) monitor and quantify SDG impacts ex-post using a method recognised by a CCP-Eligible program; (2) demonstrate a quantified positive impact … |
| How are transition projects treated in terms of crediting cycles when they move to the Gold Standard for Global Goals? | They retain their existing crediting cycle and maximum crediting periods … |

### `scripts/03_unanswerable.py`: questions the documents do *not* answer
- From 60 random seed passages, `gpt-oss-20b` writes 4 realistic analyst questions each:
  - **missing_detail**: same topic, but asks for a fee, number, deadline or step the corpus doesn't give
  - **out_of_scope**: a neighbouring programme or jurisdiction not in the corpus (e.g. CORSIA)
  - **false_premise**: confidently assumes a rule or number that isn't stated
- **Verification:** each candidate is run through the *real* retriever (top-4, exactly as at test time).
  A judge must say the sources do **not** answer it. "yes" and "partial" are discarded.
- 40% go to test. Training questions that are near-duplicates of a test question (cosine ≥ 0.88) are removed.

### `scripts/04_build_sets.py`: training set + test set
Every training example looks exactly like inference: **system prompt + 4 retrieved sources + question**.
- **Positive:** gold passage among the sources (plus 3 retrieved distractors) → answer + `[S#]` citation.
- **Hard negative (contrastive):** the *same* question (~35% of positives) with its evidence removed and
  the other retrieved passages kept → `Not found in the provided documents.`
- **Unanswerable:** verified out-of-corpus question with its real retrieved sources → refusal.
- Only **train-split** chunks are ever used as training context (verified by a leakage test).
- A regex drops leftover "the passage / this section" questions.
- Outputs: `outputs/sft/train.jsonl`, `outputs/sft/validation.jsonl` (AlignTune-ready `messages`
  format) and **`outputs/test_candidates.csv`**, which a human checks row by row (`keep=0` to drop).

### `scripts/05_train.py`: AlignTune fine-tuning
- `aligntune.create_sft_trainer` on `unsloth/Qwen3-4B-Instruct-2507`
- **LoRA** r=16, α=32, dropout 0.05 · **4-bit** quantisation · fp16 · max sequence 3072 tokens
- batch 2 × grad-accum 8 (effective 16) · 2 epochs · lr 2e-4
- `--max-samples 40 --epochs 1` gives a 5-minute smoke test. `--backend trl` and `--engine trl` are fallbacks.
- The LoRA adapter is saved to `outputs/adapter/`.

### `scripts/06_generate.py`: the three systems
All three see **the same question and the same top-4 retrieved passages**:

| System | Model | Prompt |
|---|---|---|
| `base_rag` | Qwen3-4B (base) | simple "use the context" prompt |
| `base_strict` | Qwen3-4B (base) | the strict grounding prompt we train with (**strong baseline**) |
| `tuned_rag` | Qwen3-4B + our LoRA | strict grounding prompt |

The adapter is toggled on and off in one loaded model, so the base and tuned systems share identical weights.
It also records whether retrieval actually found the gold passage.

### `scripts/07_evaluate.py`: scoring (Groq judge + AuditKIT)
See [Evaluation](#evaluation-how-fewer-hallucinations-is-measured). Writes `outputs/results/metrics.json`,
`report.md` and `chart.png`.

### `app.py`: demo
A Gradio app with **base (strict prompt)** and **fine-tuned** side by side on the same question.
It shows the retrieved sources and highlights the ones each answer cites. `share=True` gives a public link.

## Key design decisions
| Decision | Why |
|---|---|
| **Contrastive hard negatives** (same question with and without evidence) | Teaches the model to check *whether the answer is in the sources*, not *whether the topic sounds familiar*. This is the core trick. |
| **Strict-prompt baseline** | Gains must come from fine-tuning, not from a better prompt. Base and tuned get the identical prompt. |
| **Exact refusal string** | `Not found in the provided documents.` can be detected automatically, so refusal metrics need no judge. |
| **Different judge family from generator** | gpt-oss writes the QA pairs and Qwen gates them, which reduces self-agreement bias. |
| **Block-level split + buffer chunks** | Overlapping chunks can't leak test passages into training. |
| **Verified unanswerables** | A "not in the docs" question is kept only if real retrieval confirms the answer isn't there. |
| **Hybrid retrieval** (BGE-small dense + BM25, reciprocal-rank fusion) | Dense finds paraphrases, BM25 finds exact terms like "VM0048" or "Article 6.2". |
| **Over-refusal tracked separately** | A model that always says "not found" would score 0% hallucination. We report over-refusal too, including only the cases where evidence *was* retrieved. |
| **Free-tier engineering** | A disk cache for every LLM call, backoff that reads Groq's "try again in Xs", and per-model task split under 1000 req/day / 8000 TPM. |

## Evaluation: how "fewer hallucinations" is measured
Test set: hand-checked, about half answerable and half not in the documents, built only from test-split chunks.

| Metric | Definition |
|---|---|
| **Hallucination rate** ↓ | % of responses with ≥1 claim not supported by the retrieved sources (judge: `gpt-oss-120b`), with a **95% bootstrap CI** |
| **Correct refusal** ↑ | % of unanswerable questions answered with the refusal sentence |
| **Over-refusal** ↓ | % of answerable questions refused, also reported only where the gold passage *was* retrieved |
| **Accuracy** ↑ | answer matches the gold answer (LLM judge) |
| **Citation rate** ↑ | % of answers that cite a source `[S#]` |
| **Retrieval recall** | % of answerable questions where the gold (or neighbouring) chunk was in the top 4 |
| **AuditKIT** | `Hallucination` (claim-level, LLM judge), `lexical_groundedness`, `context_coverage`, `f1_score`, plus `ak.compare` base vs tuned |

### Results
*Pending: filled in after the Kaggle run.*

| System | Hallucination ↓ | Correct refusal ↑ | Over-refusal ↓ | Accuracy ↑ |
|---|---|---|---|---|
| Base + RAG | – | – | – | – |
| Base + strict prompt + RAG | – | – | – | – |
| **Fine-tuned + RAG** | – | – | – | – |

## Tech stack
| Area | Tools |
|---|---|
| Data curation | **CuratorKIT** 0.1.27 (QA generation, hallucination gate, MinHash dedup, cleaning) |
| Fine-tuning | **AlignTune** (SFT + LoRA), Unsloth, TRL, PEFT, bitsandbytes (4-bit) |
| Evaluation | **AuditKIT** (hallucination, groundedness, coverage, F1), custom judge + bootstrap CIs |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| LLMs for data + judging | **Qwen2.5-14B served by Ollama on the GPU**, called through LiteLLM by CuratorKIT, our scripts and AuditKIT. Groq `gpt-oss-120b` was used for the laptop runs until its free daily limit ran out. |
| Retrieval | `BAAI/bge-small-en-v1.5` (sentence-transformers) + `rank_bm25`, RRF fusion |
| PDF parsing | `pypdf` |
| Compute | Laptop CPU for ingestion; **free Colab T4 GPU** (or Kaggle) for curation, training, inference and judging |
| Demo | Gradio |

## Repository layout
```
config.py               all settings (models, paths, sizes), env-overridable
common/
  llm.py                LiteLLM wrapper: disk cache, rate-limit backoff, JSON parsing, thread pool
  prompts.py            system prompts, source formatting, refusal detection, citation parsing
  retrieval.py          hybrid dense + BM25 index with RRF, embedding cache
  model.py              loads base + LoRA adapter, batched greedy generation, adapter on/off
scripts/
  01_ingest.py … 07_evaluate.py   the pipeline (see above)
kaggle/colab_run.ipynb   one-click Colab notebook: CuratorKIT data → AlignTune training → generate → AuditKIT eval → demo
kaggle/train_eval.ipynb  the same pipeline for Kaggle
app.py                  Gradio side-by-side demo
docs/SOURCES.md         exact document versions and links
requirements.txt
.env.example            GROQ_API_KEY / HF_TOKEN template
```

## How to run

### 0. Setup
```bash
git clone https://github.com/snehapal0905/grounded-carbon-llm && cd grounded-carbon-llm
python -m venv .venv && .venv/Scripts/activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt                        # GPU-only packages are only needed on Kaggle
cp .env.example .env                                   # add GROQ_API_KEY (console.groq.com) and HF_TOKEN
```
Download the 7 PDFs listed in [`docs/SOURCES.md`](docs/SOURCES.md) into `data/raw/`.

### 1–4. Build the data (CPU is fine, about 3 h on the Groq free tier)
```bash
python scripts/01_ingest.py
python scripts/02_curate.py          # CuratorKIT
python scripts/03_unanswerable.py
python scripts/04_build_sets.py
```
Then open `outputs/test_candidates.csv`, set `keep=0` on wrong or ambiguous rows, and keep ≥75 per class.

> All LLM calls are cached in `data/work/llm_cache.jsonl`, so a re-run never pays twice. On Groq's free tier
> (1000 requests/day, 8000 tokens/min per model) the work is spread over three models.

### 5–7. Train and evaluate (Kaggle, GPU T4, Internet ON)
1. Upload this folder (with `data/work/chunks.jsonl`, `outputs/sft/` and the checked CSV) as a Kaggle Dataset.
2. Add the secrets `GROQ_API_KEY` and `HF_TOKEN`.
3. Open `kaggle/train_eval.ipynb` and **Run All**: smoke test → full training (~1 h) → generate → evaluate → zip results.

### Demo
```bash
python app.py        # prints a public gradio.live link
```

## Progress
- [x] Corpus chosen and downloaded (7 official documents)
- [x] Ingestion and leakage-safe split (445 chunks)
- [x] Full pipeline code: curation, unanswerables, set building, training, generation, evaluation, demo
- [x] CuratorKIT smoke run (36/36 QA pairs passed the hallucination gate)
- [x] Leakage test (no test chunk appears in any training context)
- [x] QA generation for 183 of 260 chunks with Groq gpt-oss-120b (~550 pairs, cached in `data/work/llm_cache.jsonl`)
- [x] Moved the GPU and LLM work to Colab with a local Qwen2.5-14B (see *Engineering notes* below)
- [ ] Full **CuratorKIT** curation run of 260 chunks: generation → hallucination gate → dedup *(running on Colab)*
- [ ] Verified unanswerable questions
- [ ] SFT set + test set (LLM-verified; hand spot-check if time allows)
- [ ] Fine-tuning with **AlignTune**
- [ ] Evaluation with **AuditKIT** + results table
- [ ] Publish adapter and dataset on Hugging Face

## Engineering notes: what broke and how it was fixed
| Problem | Fix |
|---|---|
| Groq free tier: 8,000 tokens/min and **200,000 tokens/day per model**. The full pipeline needs about 2M tokens. | Every LLM call is cached on disk. The remaining work moved to a local **Qwen2.5-14B on Kaggle (Ollama)** via the same LiteLLM interface. `CACHE_FALLBACK_MODELS` reuses answers already produced by Groq, so nothing is generated twice. |
| `qwen3.8-27b` on Groq also caps output at 1,000 tokens/min. | That model was used only for short outputs (the gate), not generation. |
| The laptop (16 GB RAM) ran out of memory, and Claude Code stopped background jobs that used PyTorch alongside a browser. | Everything heavy moved to a GPU notebook, where the full **CuratorKIT** run has room to finish. |
| Kaggle GPU needs phone verification, which didn't sync in time. | The same notebook runs on **Google Colab** (free T4). It clones this repo, so nothing is uploaded by hand. |
| A stalled HTTP connection froze the run for 30+ minutes. | Every LLM call has a timeout (`LLM_TIMEOUT`) plus retries with backoff that reads Groq's "try again in Xs". |
| Windows console crashed on Unicode (cp1252). | stdout/stderr are reconfigured to UTF-8 in `config.py`. |

## Limitations
- The judge is an LLM (a different family from the generator). The test set was hand-checked, but not every judge verdict.
- Retrieval recall caps answerable accuracy, so it is reported separately.
- The documents are a snapshot. Standards are revised often (VCS v5 is from Dec 2025).
- The India CCTS gazette is a third-party mirror of the official notification, English part only.

## Acknowledgements
Lexsi Labs for CuratorKIT, AlignTune and AuditKIT · Qwen team · Groq · Verra, Gold Standard, ICVCM,
UNFCCC and the Government of India for publishing their standards openly.
