"""Opt-in Qwen3-VL projection accounting. Counts shapes, not hardware events."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


def save_rank_records(output_dir, folder, records):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(output_dir) / folder
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"rank-{rank:02}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(records) + "\n", encoding="utf-8")
    tmp.replace(target)


class ProjectionAudit:
    """Count original and replayed projection calls in separate windows."""

    def __init__(self):
        self.attached = False
        self.phase = None
        self.records = []

    def attach(self, model):
        fused = set()
        modules = tuple(model.named_modules())
        for name, module in modules:
            cls = type(module)
            if cls.__name__ not in ("Qwen3VLTextMLP", "Qwen3VLTextAttention"):
                continue
            impl = sys.modules[cls.__module__]
            if cls.__name__ == "Qwen3VLTextMLP" and impl._FUSED_GATEUP and impl._HAS_NPU_SWIGLU:
                parts = (module.gate_proj, module.up_proj)
            elif cls.__name__ == "Qwen3VLTextAttention" and impl._FUSED_QKV and impl._npu is not None:
                parts = (module.q_proj, module.k_proj, module.v_proj)
            else:
                continue
            fused.update(parts)
            module.register_forward_pre_hook(self._fused_hook(parts), with_kwargs=True)
        for name, module in modules:
            if not isinstance(module, nn.Linear):
                continue
            path = name.split(".")
            group = ("lm_head" if name == "lm_head" else "vision" if "visual" in path
                     else "text" if "language_model" in path else "ross_other")
            module.register_forward_pre_hook(self._linear_hook(group, module in fused), with_kwargs=True)
        self.attached = True

    def _linear_hook(self, group, fused_child):
        def hook(module, args, kwargs):
            if self.phase is None:
                return
            if fused_child:
                raise RuntimeError("Projection audit: fused projection also called its Linear child")
            self._record(group, module, args[0] if args else kwargs["input"], fused=False)
        return hook

    def _fused_hook(self, parts):
        def hook(module, args, kwargs):
            if self.phase is None:
                return
            x = args[0] if args else kwargs["hidden_states" if "hidden_states" in kwargs else "x"]
            for part in parts:
                self._record("text", part, x, fused=True)
        return hook

    def _record(self, group, layer, x, fused):
        rows = math.prod(x.shape[:-1])
        flops = 2 * rows * layer.in_features * layer.out_features
        if self.phase == "replay":
            self.counts[group]["replay"] += flops
        else:
            key = "trainable" if layer.weight.requires_grad else "frozen"
            self.counts[group][key] += flops
            if fused:
                self.counts[group]["fused_forward"] += flops
            if group == "lm_head":
                self.counts[group]["rows"] += rows

    def begin_forward(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.phase = "forward"

    def end_forward(self):
        self.phase = "replay"

    def end_step(self, step, samples):
        self.phase = None
        self.records.append({"step": step, "samples": samples, "groups": self.counts})

    def save(self, output_dir):
        save_rank_records(output_dir, "projection-audit", self.records)
