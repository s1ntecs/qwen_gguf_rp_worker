"""ComfyUI node: preload built-in adapters once, apply request-specific strengths."""
from __future__ import annotations

import logging
import time

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import folder_paths

from lora_catalog import BUILTIN_LORAS, validate_strength

logger = logging.getLogger(__name__)


def _preload():
    cache = {}
    device = comfy.model_management.get_torch_device()
    for asset in BUILTIN_LORAS:
        started = time.monotonic()
        path = folder_paths.get_full_path_or_raise("loras", asset.filename)
        weights, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True,
        )
        # Materialize all adapters on the compute device before readiness,
        # including disabled ones. CPU is used only for CPU smoke tests.
        weights = {key: tensor.to(device=device, copy=True)
                   for key, tensor in weights.items()}
        cache[asset.key] = (weights, metadata)
        logger.info("Qwen LoRA preloaded: %s (%s) device=%s, %.3fs", asset.key,
                    asset.filename, device, time.monotonic() - started)
    return cache


# This process-wide cache survives node invalidation and new node instances.
# Loading happens when ComfyUI imports the node, before the worker is ready.
_WEIGHTS = _preload()


class QwenBuiltinLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            **{f"{asset.key}_strength": ("FLOAT", {
                "default": asset.default_strength, "min": -4.0, "max": 4.0,
                "step": 0.05,
            }) for asset in BUILTIN_LORAS},
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Qwen/LoRA"

    def apply(self, model, nsfw_strength=1.0, penis_strength=0.0, vagina_strength=0.0):
        strengths = {
            "nsfw": validate_strength(nsfw_strength),
            "penis": validate_strength(penis_strength),
            "vagina": validate_strength(vagina_strength),
        }
        result = model
        for asset in BUILTIN_LORAS:
            strength = strengths[asset.key]
            if strength == 0:
                continue
            weights, metadata = _WEIGHTS[asset.key]
            key_map = comfy.lora.model_lora_keys_unet(result.model, {})
            patches = comfy.lora.load_lora(
                comfy.lora_convert.convert_lora(dict(weights)), key_map,
            )
            if not patches:
                raise RuntimeError(f"Built-in LoRA '{asset.key}' matches no model layers")
            # Clone the patcher, retaining GGUF's patcher subclass. Never merge
            # into base weights or modify a patcher from a previous request.
            updated = result.clone()
            accepted = updated.add_patches(patches, strength)
            if len(accepted) != len(patches):
                raise RuntimeError(f"Built-in LoRA '{asset.key}' has unsupported model patches")
            if metadata:
                updated.set_attachments("lora_metadata", metadata)
            logger.info("Qwen LoRA applied from device cache: %s strength=%s matched=%d",
                        asset.key, strength, len(accepted))
            result = updated
        return (result,)


NODE_CLASS_MAPPINGS = {"QwenBuiltinLoraStack": QwenBuiltinLoraStack}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenBuiltinLoraStack": "Qwen Built-in LoRAs (cached)"}
