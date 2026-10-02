"""Exercise our cache and patch isolation at the ComfyUI API boundary."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Model:
    def __init__(self, patches=None, accepted=True):
        self.model = self
        self.patches = list(patches or [])
        self.accepted = accepted
        self.attachments = {}

    def clone(self):
        return Model(self.patches, self.accepted)

    def add_patches(self, patches, strength):
        if not self.accepted:
            return []
        self.patches.extend((key, strength) for key in patches)
        return list(patches)

    def set_attachments(self, key, value):
        self.attachments[key] = value


class LoraManagerTests(unittest.TestCase):
    def setUp(self):
        # GPU/model libraries are external to this repository. These doubles
        # preserve the ComfyUI load/convert/clone/add_patches contracts.
        self.loads = []
        self.materializations = 0
        self.devices = []
        comfy = types.ModuleType("comfy")
        comfy.utils = types.ModuleType("comfy.utils")
        comfy.utils.load_torch_file = self.load_file
        comfy.lora = types.ModuleType("comfy.lora")
        comfy.lora.model_lora_keys_unet = lambda model, keys: {"target": "target"}
        comfy.lora.load_lora = lambda weights, keys: weights
        comfy.lora_convert = types.ModuleType("comfy.lora_convert")
        comfy.lora_convert.convert_lora = lambda weights: weights
        comfy.model_management = types.ModuleType("comfy.model_management")
        comfy.model_management.get_torch_device = lambda: "cuda:0"
        folders = types.ModuleType("folder_paths")
        folders.get_full_path_or_raise = lambda category, name: f"/models/{category}/{name}"
        self.dependencies = patch.dict(sys.modules, {
            "comfy": comfy, "comfy.utils": comfy.utils, "comfy.lora": comfy.lora,
            "comfy.lora_convert": comfy.lora_convert, "folder_paths": folders,
            "comfy.model_management": comfy.model_management,
        })
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        path = Path(__file__).resolve().parents[1] / "custom_nodes" / "qwen_lora_manager.py"
        spec = importlib.util.spec_from_file_location("qwen_lora_manager_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def load_file(self, path, *, safe_load, return_metadata):
        if not safe_load or not return_metadata:
            raise AssertionError("Must use safe loading with metadata")
        self.loads.append(path)
        fixture = self

        class Tensor:
            def to(self, *, device, copy):
                if not copy:
                    raise AssertionError("Cache must own its tensors")
                fixture.materializations += 1
                fixture.devices.append(device)
                return self

        return {Path(path).name: Tensor()}, {"source": Path(path).name}

    def test_preload_materializes_all_adapters_on_compute_device(self):
        self.assertEqual(self.materializations, 3)
        self.assertEqual(self.devices, ["cuda:0"] * 3)

    def test_preloads_every_file_once_and_reuses_across_node_instances(self):
        self.assertEqual(len(self.loads), 3)
        node = self.module.QwenBuiltinLoraStack()
        first, = node.apply(Model(), 1, 0, 0)
        second, = self.module.QwenBuiltinLoraStack().apply(Model(), 0, 0.65, 0.4)
        self.assertEqual(first.patches, [("NSFW Qwen by TheseAlpacas V2.safetensors", 1.0)])
        self.assertEqual(second.patches, [
            ("qwen-image-2.1_penis_coachbate_preview1.safetensors", 0.65),
            ("qwen21_v2_000002750.safetensors", 0.4),
        ])
        self.assertEqual(len(self.loads), 3)

    def test_disabled_loras_return_original_model_without_patches(self):
        base = Model([("turbo", 1)])
        result, = self.module.QwenBuiltinLoraStack().apply(base, 0, 0, 0)
        self.assertIs(result, base)
        self.assertEqual(base.patches, [("turbo", 1)])

    def test_strength_changes_do_not_modify_base_or_previous_request(self):
        node = self.module.QwenBuiltinLoraStack()
        base = Model([("turbo", 1)])
        first, = node.apply(base, 1, 0, 0)
        second, = node.apply(base, 0.2, 0, 0)
        self.assertEqual(base.patches, [("turbo", 1)])
        self.assertEqual(first.patches[-1][1], 1.0)
        self.assertEqual(second.patches[-1][1], 0.2)
        self.assertEqual(len(second.patches), 2)

    def test_adapter_without_matching_keys_is_rejected(self):
        self.module.comfy.lora.load_lora = lambda weights, keys: {}
        with self.assertRaisesRegex(RuntimeError, "nsfw"):
            self.module.QwenBuiltinLoraStack().apply(Model(), 1, 0, 0)

    def test_unaccepted_patch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "nsfw"):
            self.module.QwenBuiltinLoraStack().apply(Model(accepted=False), 1, 0, 0)

    def test_invalid_strength_is_rejected_before_patching(self):
        base = Model()
        for value in (float("nan"), float("inf"), 4.1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.QwenBuiltinLoraStack().apply(base, value, 0, 0)
        self.assertEqual(base.patches, [])


if __name__ == "__main__":
    unittest.main()
