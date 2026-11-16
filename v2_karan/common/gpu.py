"""What GPU are we on? Picks dtype / 4-bit / batch sizes so the same code runs on a T4 and on RunPod."""
import torch


def native_bf16():
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)  # T4 only *emulates* bf16
    except TypeError:
        return torch.cuda.get_device_capability(0)[0] >= 8


def gpu_gib():
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0


BF16 = native_bf16()                 # Ampere+ (A100, L40S, 4090, H100)
BIG = gpu_gib() >= 30                 # 40-80 GB cards: no 4-bit, bigger batches
DTYPE = torch.bfloat16 if BF16 else torch.float16
DTYPE_NAME = "bfloat16" if BF16 else "float16"
GEN_BATCH = 32 if BIG else 8


def describe():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return f"{name}, {gpu_gib():.0f} GiB, dtype={DTYPE_NAME}, 4-bit={'no' if BIG else 'yes'}"
