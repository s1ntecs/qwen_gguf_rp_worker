"""Bridge the pinned GGUF mover to ComfyUI's WeightAdapter objects.

GGUF previously moves only tensors inside lists/tuples. New ComfyUI adapters
store tensors in .weights; without this bridge patch_on_device leaves Turbo
and extra LoRAs on CPU, causing transfers during every denoising step.
"""
from pathlib import Path
import sys


ORIGINAL = '''def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
'''

REPLACEMENT = '''def move_patch_to_device(item, device):
    from copy import copy
    from comfy.weight_adapter.base import WeightAdapterBase
    if isinstance(item, WeightAdapterBase):
        moved = copy(item)
        moved.weights = move_patch_to_device(item.weights, device)
        return moved
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
'''


def patch_source(source):
    source = source.replace("\r\n", "\n")
    if source.count(ORIGINAL.rstrip()) != 1:
        raise RuntimeError("GGUF adapter mover differs from pinned source")
    return source.replace(ORIGINAL.rstrip(), REPLACEMENT.rstrip(), 1)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.write_text(patch_source(path.read_text(encoding="utf-8")), encoding="utf-8")
