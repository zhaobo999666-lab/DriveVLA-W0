"""Bound Ascend ZeRO-2 gradient overflow reductions in a benchmark worker."""

import functools
import os
import runpy
import sys
from pathlib import Path

_LARGE_BYTES = 1 << 31
_CHUNK_ELEMENTS = 1 << 26


def overflow_check(original):
    """Keep DeepSpeed's NaN/Inf semantics and device-resident scalar result."""
    import torch

    announced = set()

    @functools.wraps(original)
    def check(grad):
        if grad.device.type != "npu" or grad.layout != torch.strided:
            return original(grad)
        size = grad.numel() * grad.element_size()
        if grad.numel() == 1 and grad.untyped_storage().nbytes() > size:
            # The NPU singleton reduction may dispatch a copy of the backing
            # storage rather than the element addressed by the tensor view.
            if "scalar" not in announced:
                print(f"[zero2-overflow] rank={os.environ.get('RANK', 'probe')} checking "
                      f"scalar view at offset {grad.storage_offset()} in "
                      f"{grad.untyped_storage().nbytes()} bytes of storage", flush=True)
                announced.add("scalar")
            return torch.isfinite(grad.view(())).logical_not()
        if size < _LARGE_BYTES:
            return original(grad)
        if "chunks" not in announced:
            rank = os.environ.get("RANK", "probe")
            print(f"[zero2-overflow] rank={rank} checking {size} "
                  f"bytes in chunks of at most {_CHUNK_ELEMENTS} elements", flush=True)
            announced.add("chunks")

        def bounded_parts(x):
            if x.numel() <= _CHUNK_ELEMENTS:
                yield x
            elif x.is_contiguous():
                yield from x.view(-1).split(_CHUNK_ELEMENTS)
            else:
                dim = max(range(x.ndim), key=x.shape.__getitem__)
                for part in x.split((x.size(dim) + 1) // 2, dim=dim):
                    yield from bounded_parts(part)

        return torch.stack([original(part) for part in bounded_parts(grad)]).any()

    return check


def main():
    from deepspeed.runtime.zero import stage_1_and_2

    if len(sys.argv) < 2:
        sys.exit("usage: npu_zero2_compat.py /path/to/train_qwen_vla.py [args...]")
    stage_1_and_2.has_inf_or_nan = overflow_check(stage_1_and_2.has_inf_or_nan)
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
