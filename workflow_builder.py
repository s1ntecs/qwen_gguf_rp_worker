from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable

from lora_catalog import resolve_lora_settings, validate_strength


MAX_REFERENCE_IMAGES = 10
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 25
DEFAULT_CFG = 1.0
TURBO_STEPS = 6
TURBO_SIGMAS = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"


@dataclass(frozen=True)
class ModelConfig:
    model_file: str = "qwen-image-2.1-Q4_K_M.gguf"
    text_encoder_file: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae_file: str = "qwen_image_2.1_vae_bf16.safetensors"
    turbo_lora_file: str = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"


def round_dimension(value: int | float | str | None, default: int) -> int:
    if value is None:
        value = default
    value = int(value)
    if value < 256 or value > 4096:
        raise ValueError("width/height must be between 256 and 4096")
    return max(256, min(4096, round(value / 32) * 32))


def safe_lora_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("LoRA name must be a non-empty string")
    normalized = name.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe LoRA path: {name!r}")
    return normalized


def normalize_loras(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, "", []):
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("'loras' must be a list, string, or object")

    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = safe_lora_name(item)
            strength = 1.0
        elif isinstance(item, dict):
            name = safe_lora_name(item.get("name") or item.get("file") or "")
            strength = validate_strength(item.get("strength", item.get("scale", 1.0)))
        else:
            raise ValueError("Each LoRA must be a filename or an object")

        result.append({"name": name, "strength": strength})
    return result


def _image_nodes(
    image_names: Iterable[str],
    text_inputs: dict[str, Any],
    width: int | None,
    height: int | None,
) -> tuple[dict[str, Any], list[str]]:
    graph: dict[str, Any] = {}
    names = list(image_names)
    if len(names) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Qwen Image 2.1 supports at most {MAX_REFERENCE_IMAGES} reference images"
        )

    for index, image_name in enumerate(names, start=1):
        node_id = f"image_{index}"
        graph[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }
        source: list[Any] = [node_id, 0]

        if index == 1 and width is not None and height is not None:
            resize_id = "resize_primary"
            graph[resize_id] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": source,
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "crop": "disabled",
                },
            }
            source = [resize_id, 0]
            text_inputs["resolution"] = 0

        text_inputs[f"images.image_{index}"] = source

    if names:
        text_inputs["vae"] = ["vae", 0]
    return graph, names


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    image_names: list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG,
    seed: int = 0,
    loras: list[dict[str, Any]] | None = None,
    lora_strengths: dict[str, float] | None = None,
    turbo: bool = False,
    reference_resolution: int = 1024,
    model: ModelConfig | None = None,
    output_prefix: str = "qwen21/output",
) -> tuple[dict[str, Any], int]:
    model = model or ModelConfig()
    image_names = image_names or []
    lora_strengths, loras = resolve_lora_settings(lora_strengths, normalize_loras(loras))

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' is required")
    if len(image_names) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"At most {MAX_REFERENCE_IMAGES} images are supported")

    if not turbo:
        steps = int(steps)
        if steps < 1 or steps > 100:
            raise ValueError("'steps' must be between 1 and 100")
    else:
        steps = TURBO_STEPS

    cfg_scale = float(cfg_scale)
    if cfg_scale < 0 or cfg_scale > 20:
        raise ValueError("'cfg_scale' must be between 0 and 20")

    explicit_size = width is not None or height is not None
    if explicit_size:
        if width is None or height is None:
            raise ValueError("Provide both 'width' and 'height', or neither")
        width = round_dimension(width, DEFAULT_WIDTH)
        height = round_dimension(height, DEFAULT_HEIGHT)
    elif not image_names:
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

    reference_resolution = int(reference_resolution)
    if reference_resolution < 256 or reference_resolution > 2048:
        raise ValueError("'reference_resolution' must be between 256 and 2048")

    graph: dict[str, Any] = {
        "model": {
            "class_type": "UnetLoaderGGUFAdvanced",
            "inputs": {
                "unet_name": model.model_file,
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": True,
            },
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": model.text_encoder_file,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": model.vae_file},
        },
    }

    text_inputs: dict[str, Any] = {
        "clip": ["clip", 0],
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt or "",
        "resolution": reference_resolution,
    }
    image_graph, loaded_images = _image_nodes(
        image_names, text_inputs, width, height
    )
    graph.update(image_graph)
    graph["text"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": text_inputs,
    }

    model_ref: list[Any] = ["model", 0]
    if turbo:
        graph["turbo_lora"] = {
            "class_type": "ViggleTurboLora",
            "inputs": {
                "model": model_ref,
                "lora_name": model.turbo_lora_file,
                "strength": 1.0,
            },
        }
        model_ref = ["turbo_lora", 0]

    graph["builtin_loras"] = {
        "class_type": "QwenBuiltinLoraStack",
        "inputs": {
            "model": model_ref,
            **{f"{key}_strength": strength for key, strength in lora_strengths.items()},
        },
    }
    model_ref = ["builtin_loras", 0]

    for index, lora in enumerate(loras, start=1):
        node_id = f"lora_{index}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora["name"],
                "strength_model": float(lora["strength"]),
            },
        }
        model_ref = [node_id, 0]

    if loaded_images:
        latent_ref: list[Any] = ["text", 2]
    else:
        graph["latent"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }
        latent_ref = ["latent", 0]

    if turbo:
        graph.update(
            {
                "noise": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": int(seed)},
                },
                "sigmas": {
                    "class_type": "ViggleTurboSigmas",
                    "inputs": {
                        "latent": latent_ref,
                        "nodes": TURBO_SIGMAS,
                    },
                },
                "guider": {
                    "class_type": "BasicGuider",
                    "inputs": {
                        "model": model_ref,
                        "conditioning": ["text", 0],
                    },
                },
                "sampler_select": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "sampler": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["noise", 0],
                        "guider": ["guider", 0],
                        "sampler": ["sampler_select", 0],
                        "sigmas": ["sigmas", 0],
                        "latent_image": latent_ref,
                    },
                },
            }
        )
    else:
        graph["sampler"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
                "positive": ["text", 0],
                "negative": ["text", 1],
                "latent_image": latent_ref,
                "seed": int(seed),
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        }

    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["sampler", 0],
            "vae": ["vae", 0],
        },
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["decode", 0],
            "filename_prefix": output_prefix,
        },
    }
    return graph, steps
