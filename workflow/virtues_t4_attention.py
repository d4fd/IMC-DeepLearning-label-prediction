"""Project-owned SDPA substitute for VirTues varlen attention on Tesla T4.

Call install_t4_attention() before importing any VirTues model modules. Use the
frozen encoder in eval mode; this adapter intentionally disallows dropout.
"""

import importlib
import sys
from types import ModuleType

import torch
import torch.nn.functional as F


_LAYER = "virtues.modules.layers.attention_flashattention"


def sdpa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                          deterministic=True, **kwargs):
    """Non-causal, full attention per packed segment, preserving token order."""
    if kwargs:
        raise ValueError(f"Unsupported FlashAttention options: {sorted(kwargs)}")
    if dropout_p != 0.0:
        raise ValueError("T4 adapter supports eval-mode attention only (dropout_p=0).")
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError("Expected qkv with shape (total_tokens, 3, heads, head_dim).")
    boundaries = cu_seqlens.tolist()
    if (not boundaries or boundaries[0] != 0 or boundaries[-1] != len(qkv)
            or any(end < start or end - start > max_seqlen
                   for start, end in zip(boundaries[:-1], boundaries[1:]))):
        raise ValueError("Invalid cu_seqlens or max_seqlen for qkv.")
    q, k, v = qkv.unbind(dim=1)
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end == start:
            continue
        segments = [t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(*segments, dropout_p=0.0,
                                              is_causal=False)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)


def install_t4_attention():
    """Bind the released VirTues layer to SDPA before constructing a model.

    A temporary import shim allows loading the vendor layer without installing
    FlashAttention-2. The vendor checkout on disk is never modified.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
        raise RuntimeError("T4 adapter requires a CUDA GPU with compute capability 7.x.")
    if _LAYER not in sys.modules:
        package = ModuleType("flash_attn")
        package.__path__ = []
        interface = ModuleType("flash_attn.flash_attn_interface")
        interface.flash_attn_varlen_qkvpacked_func = sdpa_varlen_qkvpacked
        names = {"flash_attn": package,
                 "flash_attn.flash_attn_interface": interface}
        previous = {name: sys.modules.get(name) for name in names}
        try:
            sys.modules.update(names)
            layer = importlib.import_module(_LAYER)
        finally:
            for name, original in previous.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
    else:
        layer = sys.modules[_LAYER]
    layer.flash_attn_varlen_qkvpacked_func = sdpa_varlen_qkvpacked
    return layer
