import sys
import types
import unittest
from unittest.mock import patch

from scripts.patch_gguf_adapters import patch_source


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


class AdapterTransferTests(unittest.TestCase):
    def test_adapter_tensors_move_without_mutating_shared_adapter(self):
        class Tensor:
            def __init__(self, device="cpu"):
                self.device = device

            def to(self, device, non_blocking):
                return self if device == self.device else Tensor(device)

        class WeightAdapterBase:
            pass

        adapter = WeightAdapterBase()
        adapter.weights = (Tensor(), Tensor(), 16, None)
        fake_torch = types.SimpleNamespace(Tensor=Tensor)
        base = types.ModuleType("comfy.weight_adapter.base")
        base.WeightAdapterBase = WeightAdapterBase
        scope = {"torch": fake_torch}
        with patch.dict(sys.modules, {"comfy.weight_adapter.base": base}):
            exec(patch_source(ORIGINAL), scope)
            moved = scope["move_patch_to_device"]([(1, adapter)], "cuda:0")[0][1]
            again = scope["move_patch_to_device"](moved, "cuda:0")
        self.assertIsNot(moved, adapter)
        self.assertEqual(moved.weights[0].device, "cuda:0")
        self.assertEqual(moved.weights[1].device, "cuda:0")
        self.assertEqual(adapter.weights[0].device, "cpu")
        self.assertEqual(moved.weights[2:], (16, None))
        self.assertIs(again.weights[0], moved.weights[0])

    def test_unexpected_upstream_source_fails_closed(self):
        with self.assertRaises(RuntimeError):
            patch_source("def move_patch_to_device(item, device): pass")

    def test_upstream_crlf_is_supported(self):
        self.assertIn("WeightAdapterBase", patch_source(ORIGINAL.replace("\n", "\r\n")))
