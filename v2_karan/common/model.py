"""Load a base model (original or merged-SFT), optionally attach a LoRA adapter, toggle it on/off.

v2: `base` can be the merged SFT checkpoint (for the DPO adapter), and max_new_tokens is larger
because quote-then-answer outputs are longer than plain answers.
"""
import contextlib
from pathlib import Path

import torch

from common.gpu import DTYPE, GEN_BATCH
from config import BASE_MODEL


def load_causal_lm(path, **kw):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPE, **kw)


class Generator:
    def __init__(self, base=BASE_MODEL, adapter_dir=None, load_in_4bit=False):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(base))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=DTYPE)
        model = load_causal_lm(str(base), **kw)
        self.has_adapter = adapter_dir is not None and (Path(adapter_dir) / "adapter_config.json").exists()
        if self.has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        elif adapter_dir is not None:
            print(f"! No adapter at {adapter_dir}")
        self.model = model.eval()

    def _ctx(self, tuned):
        if tuned:
            if not self.has_adapter:
                raise RuntimeError("Fine-tuned system requested but no adapter loaded.")
            return contextlib.nullcontext()
        return self.model.disable_adapter() if self.has_adapter else contextlib.nullcontext()

    @torch.inference_mode()
    def generate(self, batch_messages, tuned=False, max_new_tokens=320, batch_size=GEN_BATCH):
        from tqdm import tqdm
        outs = []
        # sort by length so batches pad less; restore order after
        order = sorted(range(len(batch_messages)), key=lambda i: len(str(batch_messages[i])))
        with self._ctx(tuned):
            for s in tqdm(range(0, len(order), batch_size), desc="generate"):
                idx = order[s:s + batch_size]
                prompts = [self.tok.apply_chat_template(batch_messages[i], tokenize=False,
                                                        add_generation_prompt=True) for i in idx]
                enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
                dec = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                outs += list(zip(idx, dec))
        outs.sort()
        return [o.strip() for _, o in outs]

    def free(self):
        import gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
