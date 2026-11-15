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
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_MODEL, OUT  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MERGED_DIR = Path(os.getenv("MERGED_DIR", "/content/merged_sft"))
MAX_LEN, MAX_PROMPT, MAX_COMPLETION = 3072, 2600, 448


def mk(cls, **kw):
    """Build a TRL config, silently dropping fields this TRL version doesn't have."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = [k for k in kw if k not in names]
    if dropped:
        print(f"  ({cls.__name__}: ignoring unsupported {dropped})")
    return cls(**{k: v for k, v in kw.items() if k in names})


def load_4bit(path):
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from common.model import load_causal_lm
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(str(path), device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tok


def lora():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")


def collect_adapter(run_dir, dest):
    cfgs = [p for p in Path(run_dir).rglob("adapter_config.json")]
    if not cfgs:
        raise FileNotFoundError(f"No adapter_config.json under {run_dir}")
    final = [p for p in cfgs if "checkpoint-" not in str(p)]
    src = (final or sorted(cfgs, key=lambda p: p.stat().st_mtime))[-1].parent
    if Path(dest).exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("optimizer.pt", "*.pth", "rng_state*", "scheduler.pt"))
    print(f"Adapter {src} -> {dest}")


# ------------------------------- SFT -------------------------------
def sft_aligntune(a, data, run_dir):
    from aligntune.core.backend_factory import create_sft_trainer
    tr = create_sft_trainer(
        model_name=BASE_MODEL, dataset_name=str(data), backend="trl", task_type="sft",
        output_dir=str(run_dir), num_epochs=a.epochs, batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-4, max_seq_length=MAX_LEN, max_samples=a.max_samples,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_TARGETS,
        quantization={"load_in_4bit": True}, fp16=True, bf16=False, warmup_ratio=0.05,
        completion_only_loss=True, assistant_only_loss=False, packing=False, group_by_length=False,
        gradient_checkpointing=True, load_best_model_at_end=False, save_steps=10_000, report_to="none")
    tr.train()


def sft_trl(a, data, run_dir):
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    model, tok = load_4bit(BASE_MODEL)
    ds = load_dataset("json", data_files=str(data), split="train")
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))
    cfg = mk(SFTConfig, output_dir=str(run_dir), num_train_epochs=a.epochs, per_device_train_batch_size=1,
             gradient_accumulation_steps=16, learning_rate=2e-4, max_length=MAX_LEN, fp16=True,
             logging_steps=5, save_strategy="no", warmup_ratio=0.05, lr_scheduler_type="cosine",
             gradient_checkpointing=True, completion_only_loss=True, optim="paged_adamw_8bit",
             report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora())
    tr.train()
    tr.save_model(str(run_dir))


# ------------------------------- DPO -------------------------------
DPO_KW = dict(beta=0.5, rpo_alpha=0.2, max_length=MAX_LEN, max_prompt_length=MAX_PROMPT,
              max_completion_length=MAX_COMPLETION, truncation_mode="keep_start")


def dpo_aligntune(a, model_path, data, run_dir):
    from aligntune.core.backend_factory import create_rl_trainer
    tr = create_rl_trainer(
        model_name=str(model_path), dataset_name=str(data), algorithm="dpo", backend=a.dpo_backend,
        output_dir=str(run_dir), num_epochs=1, batch_size=1, gradient_accumulation_steps=16,
        learning_rate=1e-5, max_seq_length=MAX_LEN, max_samples=a.max_samples,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_TARGETS,
        quantization={"load_in_4bit": True}, fp16=True, bf16=False, warmup_ratio=0.1,
        gradient_checkpointing=True, report_to="none", save_steps=10_000, **DPO_KW)
    tr.train()


def dpo_trl(a, model_path, data, run_dir):
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer
    model, tok = load_4bit(model_path)
    ds = load_dataset("json", data_files=str(data), split="train")
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))
    cfg = mk(DPOConfig, output_dir=str(run_dir), num_train_epochs=1, per_device_train_batch_size=1,
             gradient_accumulation_steps=16, learning_rate=1e-5, fp16=True, logging_steps=5,
             save_strategy="no", warmup_ratio=0.1, lr_scheduler_type="cosine", gradient_checkpointing=True,
             optim="paged_adamw_8bit", report_to="none", **DPO_KW)
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

    if a.stage == "sft":
        run_dir = root / "run_sft"
        run(sft_aligntune, sft_trl, a.engine, a, OUT / "sft" / "train.jsonl", run_dir)
        collect_adapter(run_dir, root / "adapter_sft")
    elif a.stage == "merge":
        merge()
    else:
        model_path = BASE_MODEL if a.smoke else MERGED_DIR
        if not a.smoke and not (MERGED_DIR / "config.json").exists():
            sys.exit(f"{MERGED_DIR} missing - run --stage merge first")
        run_dir = root / "run_dpo"
        run(dpo_aligntune, dpo_trl, a.engine, a, model_path, OUT / "pref" / "train.jsonl", run_dir)
        collect_adapter(run_dir, root / "adapter_dpo")
    import torch
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
