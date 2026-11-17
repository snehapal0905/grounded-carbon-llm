"""Step 5 (v2) - SFT (answer-only loss) -> merge -> DPO, all through AlignTune with a plain-TRL fallback.

    python scripts/05_train.py --stage sft   [--smoke]
    python scripts/05_train.py --stage merge
    python scripts/05_train.py --stage dpo   [--smoke]

Settings follow the research report (Trust-Align 2409.11242, Ground-GRPO 2506.15522, Unsloth/TRL DPO docs):
  SFT : LoRA r16/a32 on all 7 projections, 4-bit, lr 2e-4, 2 epochs, loss on the answer only.
  DPO : starts from the MERGED SFT model (AlignTune cannot load an adapter), LoRA, 4-bit,
        beta 0.5, lr 1e-5, 1 epoch, + small NLL term on the chosen answer (rpo_alpha) so DPO does not
        drift into over-refusal; max_prompt_length raised so the sources are never truncated.
Outputs: outputs/adapter_sft/, $MERGED_DIR (local disk), outputs/adapter_dpo/
"""
import argparse
import dataclasses
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

# Must be set before torch is imported: stops the "3.5 GiB reserved but unallocated" fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# One GPU only (Kaggle has 2 T4s; TRL's DPOTrainer refuses nn.DataParallel). Must precede the torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.gpu import BF16, BIG, DTYPE, DTYPE_NAME, describe  # noqa: E402
from config import BASE_MODEL, OUT  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MERGED_DIR = Path(os.getenv("MERGED_DIR", "/content/merged_sft"))
MAX_LEN, MAX_PROMPT, MAX_COMPLETION = 3072, 2600, 448
# Pairs longer than this are skipped. TRL >= 1.18 scores only the answer tokens (fused LM head), so on a
# 16 GB T4 full-length pairs fit; on older TRL set DPO_MAX_TOKENS=2048.
DPO_MAX_TOKENS = int(os.getenv("DPO_MAX_TOKENS", str(MAX_LEN)))
SAVE_EVERY = 10  # optimizer steps between checkpoints (resume after a disconnect)
# Precision: bf16 on Ampere+ (RunPod), fp16 on T4 (it only emulates bf16 -> very slow).
# 4-bit only on small cards; on 40-80 GB cards plain bf16 LoRA is faster and a bit more accurate.
QUANT = {} if BIG else {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": DTYPE_NAME}
PREC = dict(precision="bf16" if BF16 else "fp16", fp16=not BF16, bf16=BF16)
SFT_BS, DPO_BS = (4, 2) if BIG else (1, 1)   # effective batch stays 16


def mk(cls, **kw):
    """Build a TRL config, silently dropping fields this TRL version doesn't have."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = [k for k in kw if k not in names]
    if dropped:
        print(f"  ({cls.__name__}: ignoring unsupported {dropped})")
    return cls(**{k: v for k, v in kw.items() if k in names})


def load_train_model(path):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from common.model import load_causal_lm
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if BIG:
        model = load_causal_lm(str(path), device_map={"": 0})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        return model, tok
    from peft import prepare_model_for_kbit_training
    model = load_causal_lm(str(path), device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tok


def lora():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")


def fp32_trainable():
    """For fp16 GPUs (T4). fp16 mixed precision needs fp32 trainable weights: some loaders leave LoRA in bf16,
    which crashes the grad scaler ('..._unscale_ not implemented for BFloat16'). Frozen bf16 weights
    (embeddings / lm_head / norms) go to fp16: the T4 has no bf16 matmul ('gemm input type BFloat16 ... compute
    capability 8.0')."""
    import torch
    from transformers import Trainer
    orig = Trainer.__init__

    def init(self, *args, **kw):
        orig(self, *args, **kw)
        n = m = 0
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                n += 1
            elif not p.requires_grad and p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
                m += 1
        print(f">> fp16 GPU: {n} trainable tensors -> fp32, {m} frozen bf16 tensors -> fp16")

    Trainer.__init__ = init

    # AlignTune builds DPOConfig without fp16/bf16 flags, and TRL then defaults to bf16=True (bf16 autocast),
    # which the T4 cannot run. Force fp16 on every TRL config created without an explicit precision.
    import trl
    for name in ("DPOConfig", "SFTConfig"):
        cls = getattr(trl, name, None)
        if cls is None:
            continue
        cinit = cls.__init__

        def cfg_init(self, *a, _orig=cinit, **kw):
            if not kw.get("bf16") and "fp16" not in kw:
                kw.update(fp16=True, bf16=False)
            _orig(self, *a, **kw)

        cls.__init__ = cfg_init


def enable_resume(run_dir):
    """Make every HF Trainer (AlignTune's included) resume from the newest checkpoint in run_dir."""
    from transformers import Trainer
    orig = Trainer.train

    def train(self, resume_from_checkpoint=None, *args, **kw):
        cks = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if resume_from_checkpoint is None and cks:
            resume_from_checkpoint = str(cks[-1])
            print(f">> resuming from {resume_from_checkpoint}")
        return orig(self, resume_from_checkpoint, *args, **kw)

    Trainer.train = train


def short_pairs(data, model_path, dest, limit=None):
    """Write the preference pairs that fit in DPO_MAX_TOKENS (prompt + longer answer) to dest."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path))
    rows = [json.loads(l) for l in open(data, encoding="utf-8") if l.strip()]
    keep = []
    for r in rows:
        n = max(len(tok(tok.apply_chat_template(r["prompt"] + r[k], tokenize=False),
                        add_special_tokens=False)["input_ids"]) for k in ("chosen", "rejected"))
        if n <= DPO_MAX_TOKENS:
            keep.append(r)
    if limit:
        keep = keep[:limit]
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DPO pairs: kept {len(keep)} of {len(rows)} (<= {DPO_MAX_TOKENS} tokens) -> {dest}")
    return dest


def collect_adapter(run_dir, dest):
    cfgs = [p for p in Path(run_dir).rglob("adapter_config.json")]
    if not cfgs:
        raise FileNotFoundError(f"No adapter_config.json under {run_dir}")
    # TRL's DPOTrainer keeps a frozen copy of the starting adapter named "ref"; PEFT saves it in a "ref/"
    # subfolder next to the trained "default" adapter. Never pick that copy: take the shallowest final adapter.
    cfgs = [p for p in cfgs if "ref" not in p.relative_to(run_dir).parts[:-1]]
    final = [p for p in cfgs if "checkpoint-" not in str(p)]
    if final:
        src = min(final, key=lambda p: len(p.parts)).parent
    else:
        src = sorted(cfgs, key=lambda p: int(re.search(r"checkpoint-(\d+)", str(p)).group(1)))[-1].parent
    if Path(dest).exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("optimizer.pt", "*.pth", "rng_state*", "scheduler.pt"))
    (Path(dest) / "SOURCE").write_text(f"{src}\n")   # run_all.sh checks this to reject a bad copy
    print(f"Adapter {src} -> {dest}")


# ------------------------------- SFT -------------------------------
def sft_aligntune(a, data, run_dir):
    from aligntune.core.backend_factory import create_sft_trainer
    tr = create_sft_trainer(
        model_name=BASE_MODEL, dataset_name=str(data), backend="trl", task_type="sft",
        output_dir=str(run_dir), num_epochs=a.epochs, batch_size=SFT_BS, gradient_accumulation_steps=16 // SFT_BS,
        learning_rate=2e-4, max_seq_length=MAX_LEN, max_samples=a.max_samples,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_TARGETS,
        quantization=QUANT, **PREC, warmup_ratio=0.05,
        completion_only_loss=True, assistant_only_loss=False, packing=False, group_by_length=False,
        gradient_checkpointing=True, use_gradient_checkpointing=True, load_best_model_at_end=False,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=1, logging_steps=5, report_to="none")
    tr.train()


def sft_trl(a, data, run_dir):
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    model, tok = load_train_model(BASE_MODEL)
    ds = load_dataset("json", data_files=str(data), split="train")
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))
    cfg = mk(SFTConfig, output_dir=str(run_dir), num_train_epochs=a.epochs, per_device_train_batch_size=SFT_BS,
             gradient_accumulation_steps=16 // SFT_BS, learning_rate=2e-4, max_length=MAX_LEN,
             fp16=not BF16, bf16=BF16,
             logging_steps=5, save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=1,
             warmup_ratio=0.05, lr_scheduler_type="cosine",
             gradient_checkpointing=True, completion_only_loss=True, optim="adamw_torch" if BIG else "paged_adamw_8bit",
             report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora())
    tr.train()
    tr.save_model(str(run_dir))


# ------------------------------- DPO -------------------------------
# Newer TRL dropped rpo_alpha/max_prompt_length; the NLL-on-chosen term is loss_type ["sigmoid","sft"].
# mk() drops whichever spelling the installed TRL doesn't know. Reference log-probs are computed once up
# front (precompute_ref_log_probs) so training never holds the policy and reference passes together.
DPO_KW = dict(beta=0.5, rpo_alpha=0.2, max_length=MAX_LEN, max_prompt_length=MAX_PROMPT,
              max_completion_length=MAX_COMPLETION, truncation_mode="keep_start")


def dpo_aligntune(a, model_path, data, run_dir):
    from aligntune.core.backend_factory import create_rl_trainer
    tr = create_rl_trainer(
        model_name=str(model_path), dataset_name=str(data), algorithm="dpo", backend=a.dpo_backend,
        output_dir=str(run_dir), num_epochs=1, batch_size=DPO_BS, gradient_accumulation_steps=16 // DPO_BS,
        learning_rate=1e-5, max_seq_length=MAX_LEN, max_samples=a.max_samples,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_TARGETS,
        quantization=QUANT, **PREC, warmup_ratio=0.1,
        gradient_checkpointing=True, use_gradient_checkpointing=True, report_to="none", logging_steps=5,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=1,
        loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2], **DPO_KW)
    tr.train()


def dpo_loss_kw(cls):
    """RPO-style NLL term: TRL >= 1.0 spells it as a weighted 'sft' loss, older TRL as rpo_alpha (in DPO_KW)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 0.2]} if "loss_weights" in names else {}


def dpo_trl(a, model_path, data, run_dir):
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer
    model, tok = load_train_model(model_path)
    ds = load_dataset("json", data_files=str(data), split="train")
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))
    cfg = mk(DPOConfig, output_dir=str(run_dir), num_train_epochs=1, per_device_train_batch_size=DPO_BS,
             gradient_accumulation_steps=16 // DPO_BS, learning_rate=1e-5, fp16=not BF16, bf16=BF16,
             logging_steps=5,
             save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=1, warmup_ratio=0.1,
             lr_scheduler_type="cosine", gradient_checkpointing=True,
             optim="adamw_torch" if BIG else "paged_adamw_8bit", report_to="none",
             **dpo_loss_kw(DPOConfig), **DPO_KW)
    tr = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok,
                    peft_config=lora())
    tr.train()
    tr.save_model(str(run_dir))


def run(primary, fallback, engine, *args):
    if engine == "aligntune":
        try:
            return primary(*args)
        except Exception:
            traceback.print_exc()
            print("\n!! AlignTune path failed - falling back to plain TRL (same settings).\n")
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    fallback(*args)


def merge():
    from peft import PeftModel
    from transformers import AutoTokenizer
    from common.model import load_causal_lm
    src = OUT / "adapter_sft"
    base = load_causal_lm(BASE_MODEL, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(src)).merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), max_shard_size="2GB")
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(str(MERGED_DIR))
    print(f"Merged SFT model -> {MERGED_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "merge", "dpo"], required=True)
    ap.add_argument("--engine", choices=["aligntune", "trl"], default="aligntune")
    ap.add_argument("--dpo-backend", choices=["unsloth", "trl"], default="trl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run into outputs/smoke/ to test the setup")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = OUT / "smoke" if a.smoke else OUT
    if a.smoke:
        a.max_samples, a.epochs = a.max_samples or 24, 1
        for d in ("run_sft", "run_dpo"):
            shutil.rmtree(root / d, ignore_errors=True)
    print("GPU:", describe())
    if not BF16:
        fp32_trainable()

    if a.stage == "sft":
        run_dir = root / "run_sft"
        enable_resume(run_dir)
        run(sft_aligntune, sft_trl, a.engine, a, OUT / "sft" / "train.jsonl", run_dir)
        collect_adapter(run_dir, root / "adapter_sft")
    elif a.stage == "merge":
        merge()
    else:
        model_path = BASE_MODEL if a.smoke else MERGED_DIR
        if not a.smoke and not (MERGED_DIR / "config.json").exists():
            sys.exit(f"{MERGED_DIR} missing - run --stage merge first")
        run_dir = root / "run_dpo"
        enable_resume(run_dir)
        data = short_pairs(OUT / "pref" / "train.jsonl", model_path, root / "pref_dpo.jsonl", a.max_samples)
        run(dpo_aligntune, dpo_trl, a.engine, a, model_path, data, run_dir)
        collect_adapter(run_dir, root / "adapter_dpo")
    import torch
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
