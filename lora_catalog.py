"""Pinned built-in adapters and shared request validation (no GPU dependencies)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LORA_REPO = "sintecs/Qwen2.1_loras"
LORA_REVISION = "9181d64c28d06ce05425c741a2e551c0d812ae41"


@dataclass(frozen=True)
class LoraAsset:
    key: str
    filename: str
    sha256: str
    default_strength: float


BUILTIN_LORAS = (
    LoraAsset("nsfw", "NSFW Qwen by TheseAlpacas V2.safetensors",
              "83aba822ad8ee10a1e836828f5088259d93bce7b5f3419b46dbc0258c2be46a9", 1.0),
    LoraAsset("penis", "qwen-image-2.1_penis_coachbate_preview1.safetensors",
              "0f79fae880ba2b89960676e036dfee1fea6020f7acbf6d1a333dc14b477cdeeb", 0.0),
    LoraAsset("vagina", "qwen21_v2_000002750.safetensors",
              "71d8edbfbb693da3b1acdfe3a48c1c61942974cb1dfa864e5f0862630a0a3c4c", 0.0),
)


def validate_strength(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("LoRA strength must be a finite number between -4 and 4")
    try:
        strength = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LoRA strength must be a finite number between -4 and 4") from exc
    if not math.isfinite(strength) or not -4.0 <= strength <= 4.0:
        raise ValueError("LoRA strength must be a finite number between -4 and 4")
    return strength


def resolve_lora_settings(
    raw_strengths: Any,
    loras: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Resolve defaults and migrate legacy filename entries without double patches."""
    if raw_strengths is None:
        raw_strengths = {}
    if not isinstance(raw_strengths, dict):
        raise ValueError("'lora_strengths' must be an object")
    strengths = {asset.key: asset.default_strength for asset in BUILTIN_LORAS}
    unknown = set(raw_strengths) - strengths.keys()
    if unknown:
        raise ValueError(f"Unknown built-in LoRA keys: {', '.join(map(str, unknown))}")
    strengths.update({key: validate_strength(value) for key, value in raw_strengths.items()})
    by_filename = {asset.filename: asset.key for asset in BUILTIN_LORAS}
    extras = []
    seen = set()
    for lora in loras:
        key = by_filename.get(lora["name"])
        if key is None:
            extras.append(lora)
            continue
        if key in seen:
            raise ValueError(f"Built-in LoRA '{key}' was supplied more than once")
        seen.add(key)
        strength = validate_strength(lora["strength"])
        if key in raw_strengths and strengths[key] != strength:
            raise ValueError(f"Conflicting strengths for built-in LoRA '{key}'")
        strengths[key] = strength
    return strengths, extras
