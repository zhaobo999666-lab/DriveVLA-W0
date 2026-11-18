"""Reuse FA2 padding metadata for one training microbatch, including GC replay."""

from contextlib import nullcontext

import torch


class FlashUnpadCache:
    """Scope the override to one synchronous Trainer.training_step per process."""

    def __init__(self, flash_utils, *, enabled, markers=False):
        self.flash_utils = flash_utils
        self.enabled = enabled
        self.markers = markers
        self.hits = 0
        self.builds = 0
        self._cache = {}

    def __enter__(self):
        self._original = self.flash_utils._get_unpad_data
        self.flash_utils._get_unpad_data = self._get
        return self

    def __exit__(self, *exc):
        self.flash_utils._get_unpad_data = self._original
        self._cache.clear()

    @staticmethod
    def _key(mask):
        if (mask.ndim != 2 or not mask.is_contiguous() or mask.numel() == 0
                or mask.dtype not in (torch.bool, torch.uint8, torch.int32, torch.int64)):
            return None
        try:
            # Qwen3-VL passes the mask as a checkpoint keyword argument. The
            # same Tensor object is visible on replay. Do not alias other views.
            return id(mask), mask._version
        except RuntimeError:
            return None

    def _get(self, mask):
        key = self._key(mask) if self.enabled else None
        if key is not None and key in self._cache:
            self.hits += 1
            return self._cache[key][1]
        scope = torch.profiler.record_function("DriveVLA::FA2_unpad_metadata") if self.markers else nullcontext()
        with scope:
            result = self._original(mask)
        self.builds += 1
        if key is not None and len(self._cache) < 4:
            self._cache[key] = (mask, result)
        return result
