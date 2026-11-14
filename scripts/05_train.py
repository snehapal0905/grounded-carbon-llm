"""Step 5 - fine-tune with AlignTune (LoRA SFT, Unsloth backend). Run on a Kaggle GPU.

    python scripts/05_train.py                       # AlignTune + Unsloth (default)
    python scripts/05_train.py --backend trl         # AlignTune + TRL backend
    python scripts/05_train.py --engine trl          # plain TRL, only if AlignTune breaks

AlignTune reads outputs/sft/train.jsonl (a `messages` column), runs it through its
CuratorKIT schema gate and applies the model's own chat template, so training and
inference use the identical prompt layout. The LoRA adapter ends up in outputs/adapter/.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ADAPTER_DIR, BASE_MODEL, OUT, TRAIN_MODEL  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_aligntune(args, train_file, run_dir):
    from aligntune.core.backend_factory import create_sft_trainer

    trainer = create_sft_trainer(
        model_name=TRAIN_MODEL if args.backend == "unsloth" else BASE_MODEL,
        dataset_name=str(train_file),
        backend=args.backend,
        task_type="sft",
        output_dir=str(run_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        max_samples=args.max_samples,
        use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lora_target_modules=LORA_TARGETS,
        quantization={"load_in_4bit": True},
        fp16=True, bf16=False,                       # T4/P100 have no bf16
        warmup_ratio=0.05,
        load_best_model_at_end=False,
        save_steps=100,
        report_to="none",
    )
    result = trainer.train()
    print(result if isinstance(result, dict) else "training finished")


def train_trl(args, train_file, run_dir):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    ds = load_dataset("json", data_files=str(train_file), split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    cfg = SFTConfig(output_dir=str(run_dir), num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
                    max_length=args.max_seq_len, fp16=True, logging_steps=10, save_steps=100,
                    warmup_ratio=0.05, lr_scheduler_type="cosine", gradient_checkpointing=True,
                    assistant_only_loss=False, report_to="none")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    tr.train()
    tr.save_model(str(run_dir))
    tok.save_pretrained(str(run_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["aligntune", "trl"], default="aligntune")
    ap.add_argument("--backend", choices=["unsloth", "trl"], default="unsloth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--max-samples", type=int, default=None, help="e.g. 40 for a 5-minute smoke test")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Unsloth trains on one GPU
    train_file = OUT / "sft" / "train.jsonl"
    run_dir = OUT / "run"
    (train_aligntune if args.engine == "aligntune" else train_trl)(args, train_file, run_dir)

    # Locate the saved LoRA adapter and copy it to outputs/adapter
    cfgs = sorted(run_dir.rglob("adapter_config.json"), key=lambda p: p.stat().st_mtime)
    if not cfgs:
        sys.exit(f"No adapter_config.json under {run_dir} - check the training log.")
    src = cfgs[-1].parent
    for p in run_dir.rglob("adapter_config.json"):  # prefer the final (non-checkpoint) save
        if "checkpoint-" not in str(p):
            src = p.parent
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    shutil.copytree(src, ADAPTER_DIR, ignore=shutil.ignore_patterns("optimizer.pt", "*.pth", "rng_state*"))
    print(f"Adapter copied {src} -> {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
