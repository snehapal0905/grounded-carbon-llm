"""Load base model once, attach the LoRA adapter, and switch it on/off per system.

One copy of the weights serves both "base" and "fine-tuned" (peft.disable_adapter),
so the side-by-side comparison fits on a single 16 GB T4.
"""
import contextlib

import torch

from config import ADAPTER_DIR, BASE_MODEL


class Generator:
    def __init__(self, adapter_dir=ADAPTER_DIR, load_in_4bit=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"torch_dtype": torch.float16, "device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)
        self.has_adapter = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
        if self.has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        else:
            print(f"! No adapter at {adapter_dir}; only base-model systems are available.")
        self.model = model.eval()

    def _ctx(self, tuned):
        if tuned:
            if not self.has_adapter:
                raise RuntimeError("Fine-tuned system requested but no adapter loaded.")
            return contextlib.nullcontext()
        return self.model.disable_adapter() if self.has_adapter else contextlib.nullcontext()

    @torch.inference_mode()
    def generate(self, batch_messages, tuned=False, max_new_tokens=256, batch_size=6):
        outs = []
        with self._ctx(tuned):
            for i in range(0, len(batch_messages), batch_size):
                prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                           for m in batch_messages[i:i + batch_size]]
                enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
                outs += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [o.strip() for o in outs]
