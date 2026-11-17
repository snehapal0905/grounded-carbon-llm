#!/usr/bin/env bash
# Karan v2 pipeline as one script: Kaggle (free, see kaggle/karan_v2_kaggle.ipynb) or a RunPod pod.
# Runs on the server, so closing the laptop does not stop it. Every stage is skipped if its output
# already exists -> just rerun to resume.
#
# RunPod:  cd /workspace && git clone -b karan-v2 https://github.com/snehapal0905/grounded-carbon-llm proj
#          nohup bash proj/run_all.sh > /workspace/run.log 2>&1 &  ;  tail -f /workspace/run.log
# Env: WORKDIR (results, default /workspace)  CACHE_DIR (models, default WORKDIR)  PROJ (repo checkout)
#      DATA_SRC (folder holding data/ + outputs/ from an earlier run, copied in once)
#      STOP_AFTER=sft | train (stop after SFT / after DPO)  EPOCHS (SFT epochs, default 2)
set -euo pipefail

W0=${WORKDIR:-/workspace}
C=${CACHE_DIR:-$W0}
SAVE=$W0/lexsi_karan_v2          # same layout as the Colab Drive folder
PROJ=${PROJ:-$(cd "$(dirname "$0")" && pwd)}
W=$SAVE/data/work; O=$SAVE/outputs
mkdir -p "$W" "$O" "$C"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # one GPU (Kaggle T4 x2: DPO refuses DataParallel)
export HF_HOME=$C/hf OLLAMA_MODELS=$C/ollama PIP_CACHE_DIR=$C/pip
log() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# ---- 0. data from an earlier run (optional) ---------------------------------------------------------
if [ ! -f "$O/pref/train.jsonl" ] && [ -n "${DATA_SRC:-}" ] && [ -d "$DATA_SRC" ]; then
  log "copying earlier data/outputs from $DATA_SRC"
  cp -r "$DATA_SRC"/. "$SAVE"/
  rm -rf "$O/smoke"; [ -f "$O/adapter_sft/adapter_config.json" ] || rm -rf "$O/run_sft"
fi
if [ ! -f "$O/pref/train.jsonl" ] && ls $W0/lexsi_karan_v2*.zip >/dev/null 2>&1; then
  log "unzipping Colab data"
  for z in $W0/lexsi_karan_v2*.zip; do unzip -q -o "$z" -d $W0; done
  rm -rf "$O/smoke" "$O/run_sft" "$O/run_dpo"
fi

cd "$PROJ"; git pull -q || true
for f in chunks.jsonl llm_cache.jsonl ingest_stats.json; do
  [ -f "$W/$f" ] || cp "data/work/$f" "$W/$f"
done

# ---- 1. install -----------------------------------------------------------------------------------
if [ ! -f $C/.installed ]; then
  log "pip install"
  pip install -q aligntune rank_bm25 sentence-transformers litellm python-dotenv auditkit "curatorkit[all]" 2>&1 | tail -3
  # TRL 1.18 scores only answer tokens in DPO/SFT (no 152k-vocab logits) -> fits a 16 GB T4. AlignTune pins 1.7.1.
  pip install -q "trl==1.18.0" 2>&1 | tail -2
  apt-get -qq update >/dev/null && apt-get -qq install -y zstd pciutils lshw zip >/dev/null
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
  touch $C/.installed
fi
python -c "import torch,trl,transformers;from common.gpu import describe;print(describe(),'trl',trl.__version__,'transformers',transformers.__version__)"
# on a big GPU, regenerate every system's answers here (fair comparison); T4 answers from Colab are reused on T4
if python -c "from common.gpu import BIG; raise SystemExit(0 if BIG else 1)" && [ -d "$O/preds" ] && [ ! -f "$O/.big_gpu" ]; then
  log "moving T4 preds/results aside (regenerated on this GPU)"
  mv "$O/preds" "$O/preds_t4"; [ -d "$O/results" ] && mv "$O/results" "$O/results_t4"
fi
python -c "from common.gpu import BIG; raise SystemExit(0 if BIG else 1)" && touch "$O/.big_gpu" || true

GEN=ollama_chat/qwen2.5:14b; JUDGE=ollama_chat/gemma3:12b
export DATA_DIR=$SAVE/data OUT_DIR=$O MERGED_DIR=$C/merged_sft LITELLM_LOCAL_MODEL_COST_MAP=True \
  GEN_MODEL=$GEN GATE_MODEL=$GEN VERIFY_MODEL=$GEN JUDGE_MODEL=$JUDGE AUDIT_JUDGE=litellm:$JUDGE \
  CACHE_FALLBACK_MODELS=groq/openai/gpt-oss-120b LLM_CONCURRENCY=8 LLM_TIMEOUT=600 OLLAMA_NUM_PARALLEL=8 \
  N_PREF=800 NEG_RATE=0.25 MAX_REFUSAL_SHARE=0.28

start_ollama() { (ollama serve > $W0/ollama.log 2>&1 &); for i in $(seq 60); do curl -s localhost:11434 >/dev/null && return; sleep 1; done; }
stop_ollama()  { pkill -f "ollama serve" || true; sleep 5; }

# ---- 2. data (skipped when uploaded from Colab) ----------------------------------------------------
if [ ! -f "$O/pref/train.jsonl" ]; then
  log "building data with qwen2.5:14b"
  start_ollama; ollama pull qwen2.5:14b | tail -1
  [ -f "$W/qa_raw.jsonl" ]        || python scripts/02_curate.py --engine fallback
  [ -f "$W/unanswerable.jsonl" ]  || python scripts/03_unanswerable.py
  python scripts/04_build_sets.py
  stop_ollama
fi
cat "$W/sft_stats_v2.json" || true

# ---- 3. train ---------------------------------------------------------------------------------------
if [ ! -f "$O/smoke/adapter_dpo/adapter_config.json" ] && [ ! -f "$O/adapter_sft/adapter_config.json" ]; then
  log "smoke tests"
  python scripts/05_train.py --stage sft --smoke
  python scripts/05_train.py --stage dpo --smoke
fi
[ -f "$O/adapter_sft/adapter_config.json" ] || { log "SFT"; python scripts/05_train.py --stage sft --epochs "${EPOCHS:-2}"; }
if [ "${STOP_AFTER:-}" = "sft" ]; then rm -rf "$O/run_sft" "$O/smoke"; log "STOP_AFTER=sft: adapter saved in $O"; exit 0; fi
if [ ! -f "$O/preds/tuned_dpo.jsonl" ] && [ ! -f "$MERGED_DIR/config.json" ]; then
  log "merge"; python scripts/05_train.py --stage merge
fi
# adapters saved before the 'ref' fix have no SOURCE file and may be TRL's untrained reference copy: redo DPO
if [ -f "$O/adapter_dpo/adapter_config.json" ] && [ ! -f "$O/adapter_dpo/SOURCE" ]; then
  log "discarding old adapter_dpo (it was TRL's untrained 'ref' copy) - DPO runs again"
  rm -rf "$O/adapter_dpo" "$O/preds/tuned_dpo.jsonl"
fi
[ -f "$O/adapter_dpo/adapter_config.json" ] || { log "DPO"; python scripts/05_train.py --stage dpo; }
rm -rf "$O/run_sft" "$O/run_dpo" "$O/smoke"   # checkpoints no longer needed once adapters exist
if [ "${STOP_AFTER:-}" = "train" ]; then log "STOP_AFTER=train: adapters saved in $O"; exit 0; fi

# ---- 4. answer the test set with 4 systems, then score --------------------------------------------
# Head-to-head with the main pipeline: answer Sneha's exact 150 test questions (same chunks and split as ours,
# none of them in our training data) and judge her fine-tuned answers with the same Gemma judge. SAME_TEST=0 skips.
EXTRA=()
if [ "${SAME_TEST:-1}" = "1" ]; then
  RAW=https://raw.githubusercontent.com/snehapal0905/grounded-carbon-llm/main/outputs
  mkdir -p "$O/main_run"
  [ -f "$O/main_run/test_candidates.csv" ] || curl -fsSL "$RAW/test_candidates.csv" -o "$O/main_run/test_candidates.csv"
  [ -f "$O/main_run/tuned_rag.jsonl" ] || curl -fsSL "$RAW/preds/tuned_rag.jsonl" -o "$O/main_run/tuned_rag.jsonl"
  export TEST_CSV="$O/main_run/test_candidates.csv"
  if [ ! -f "$O/.same_test" ]; then   # earlier preds were on our own test set: keep them aside
    [ -d "$O/preds" ] && mv "$O/preds" "$O/preds_own_test"
    [ -d "$O/results" ] && mv "$O/results" "$O/results_own_test"
    touch "$O/.same_test"
  fi
  EXTRA=(--extra "$O/main_run/tuned_rag.jsonl:main_tuned")
fi
log "generate"; python scripts/06_generate.py
if [ ! -f "$O/results/report.md" ]; then
  log "judge (gemma3:12b) + HHEM + AuditKIT"
  start_ollama; ollama pull gemma3:12b | tail -1
  python scripts/07_evaluate.py "${EXTRA[@]}"
  stop_ollama
fi
cd "$O" && zip -qr $W0/karan_v2_results.zip sft pref test_candidates.csv main_run preds results adapter_sft adapter_dpo
log "DONE -> $O/results/report.md and $W0/karan_v2_results.zip"
cat "$O/results/report.md"
