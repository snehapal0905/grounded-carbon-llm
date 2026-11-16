#!/usr/bin/env bash
# Karan v2 pipeline on a RunPod GPU pod (A100 / L40S / H100). Runs on the server, so closing the
# laptop does not stop it. Every stage is skipped if its output already exists -> just rerun to resume.
#
#   cd /workspace && git clone -b karan-v2 https://github.com/snehapal0905/grounded-carbon-llm proj
#   nohup bash proj/runpod/run_all.sh > /workspace/run.log 2>&1 &
#   tail -f /workspace/run.log
set -euo pipefail

W0=/workspace
SAVE=$W0/lexsi_karan_v2          # same layout as the Colab Drive folder
PROJ=$W0/proj
W=$SAVE/data/work; O=$SAVE/outputs
mkdir -p "$W" "$O"
export HF_HOME=$W0/hf OLLAMA_MODELS=$W0/ollama PIP_CACHE_DIR=$W0/pip
log() { echo; echo "=== $(date +%H:%M:%S)  $*"; }

# ---- 0. data from Colab (optional): upload the Drive folder zip to /workspace -------------------
if [ ! -f "$O/pref/train.jsonl" ] && ls $W0/lexsi_karan_v2*.zip >/dev/null 2>&1; then
  log "unzipping Colab data"
  for z in $W0/lexsi_karan_v2*.zip; do unzip -q -o "$z" -d $W0; done
fi
# predictions made on the T4 are kept aside: all 4 systems are regenerated on this GPU (fair comparison)
if [ -d "$O/preds" ] && [ ! -f "$O/.runpod" ]; then
  log "moving Colab T4 preds/results aside"
  mv "$O/preds" "$O/preds_colab_t4"; [ -d "$O/results" ] && mv "$O/results" "$O/results_colab_t4"
  rm -rf "$O/smoke" "$O/run_sft" "$O/run_dpo"
fi
touch "$O/.runpod"

cd "$PROJ"; git pull -q || true
for f in chunks.jsonl llm_cache.jsonl ingest_stats.json; do
  [ -f "$W/$f" ] || cp "data/work/$f" "$W/$f"
done

# ---- 1. install -----------------------------------------------------------------------------------
if [ ! -f $W0/.installed ]; then
  log "pip install"
  pip install -q aligntune rank_bm25 sentence-transformers litellm python-dotenv auditkit "curatorkit[all]" 2>&1 | tail -3
  apt-get -qq update >/dev/null && apt-get -qq install -y zstd pciutils lshw zip >/dev/null
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
  touch $W0/.installed
fi
python -c "import torch,trl,transformers;from common.gpu import describe;print(describe(),'trl',trl.__version__,'transformers',transformers.__version__)"

GEN=ollama_chat/qwen2.5:14b; JUDGE=ollama_chat/gemma3:12b
export DATA_DIR=$SAVE/data OUT_DIR=$O MERGED_DIR=$W0/merged_sft LITELLM_LOCAL_MODEL_COST_MAP=True \
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
[ -f "$O/adapter_sft/adapter_config.json" ] || { log "SFT"; python scripts/05_train.py --stage sft --epochs 2; }
if [ ! -f "$O/preds/tuned_dpo.jsonl" ] && [ ! -f "$MERGED_DIR/config.json" ]; then
  log "merge"; python scripts/05_train.py --stage merge
fi
[ -f "$O/adapter_dpo/adapter_config.json" ] || { log "DPO"; python scripts/05_train.py --stage dpo; }

# ---- 4. answer the test set with 4 systems, then score --------------------------------------------
log "generate"; python scripts/06_generate.py
if [ ! -f "$O/results/report.md" ]; then
  log "judge (gemma3:12b) + HHEM + AuditKIT"
  start_ollama; ollama pull gemma3:12b | tail -1
  python scripts/07_evaluate.py
  stop_ollama
fi
cd "$O" && zip -qr $W0/karan_v2_results.zip sft pref test_candidates.csv preds results adapter_sft adapter_dpo
log "DONE -> $O/results/report.md and $W0/karan_v2_results.zip"
cat "$O/results/report.md"
