from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import random
import socket
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import runpod
from PIL import Image, ImageOps
from runpod.serverless.modules.rp_logger import RunPodLogger

from workflow_builder import ModelConfig, build_workflow, normalize_loras

logger = RunPodLogger()

COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
COMFY_INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/comfyui/input"))
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/comfyui/output"))
COMFY_READY_TIMEOUT = int(os.getenv("COMFY_READY_TIMEOUT", "180"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "900"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "45"))
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", str(30 * 1024 * 1024)))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", "40000000"))
ALLOW_PRIVATE_IMAGE_URLS = os.getenv("ALLOW_PRIVATE_IMAGE_URLS", "false").lower() in {"1","true","yes"}

MODEL = ModelConfig(
    model_file=os.getenv("QWEN_MODEL_FILE", "qwen-image-2.1-Q4_K_M.gguf"),
    text_encoder_file=os.getenv("QWEN_TEXT_ENCODER_FILE", "qwen3vl_8b_int8_convrot.safetensors"),
    vae_file=os.getenv("QWEN_VAE_FILE", "qwen_image_2.1_vae_bf16.safetensors"),
    turbo_lora_file=os.getenv(
        "QWEN_TURBO_LORA_FILE",
        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
    ),
)

class InputError(ValueError):
    pass

def wait_for_comfy() -> None:
    deadline = time.time() + COMFY_READY_TIMEOUT
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/object_info", timeout=5)
            if r.ok:
                return
        except requests.RequestException as exc:
            last = exc
        time.sleep(0.5)
    raise RuntimeError(f"ComfyUI not ready after {COMFY_READY_TIMEOUT}s: {last}")

def validate_remote_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in {"http","https"} or not p.hostname:
        raise InputError("Only http(s) image URLs are allowed")
    if ALLOW_PRIVATE_IMAGE_URLS:
        return
    try:
        addresses = {x[4][0] for x in socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme=="https" else 80))}
    except socket.gaierror as exc:
        raise InputError(f"Could not resolve image host: {p.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%",1)[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise InputError(f"Private/local image URL is not allowed: {p.hostname}")

def download_url(url: str) -> bytes:
    current = url
    for _ in range(6):
        validate_remote_url(current)
        with requests.get(current, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=False) as r:
            if r.is_redirect or r.is_permanent_redirect:
                location = r.headers.get("location")
                if not location:
                    raise InputError("Redirect missing Location header")
                current = urljoin(current, location)
                continue
            r.raise_for_status()
            declared = r.headers.get("content-length")
            if declared and int(declared) > MAX_INPUT_BYTES:
                raise InputError("Input image too large")
            chunks, size = [], 0
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_INPUT_BYTES:
                    raise InputError("Input image too large")
                chunks.append(chunk)
            return b"".join(chunks)
    raise InputError("Too many redirects")

def decode_b64(value: str) -> bytes:
    if value.startswith("data:"):
        if "," not in value:
            raise InputError("Malformed data URI")
        value = value.split(",",1)[1]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError(f"Invalid base64 image: {exc}") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise InputError("Input image too large")
    return data

def source_bytes(source: Any) -> bytes:
    if isinstance(source, dict):
        if source.get("url") or source.get("image_url"):
            return download_url(str(source.get("url") or source.get("image_url")))
        for key in ("base64","image_base64","data"):
            if source.get(key):
                return decode_b64(str(source[key]))
        raise InputError("Image object must contain url/image_url or base64/image_base64/data")
    if not isinstance(source, str) or not source.strip():
        raise InputError("Image must be URL/base64/data URI")
    source = source.strip()
    return download_url(source) if source.startswith(("http://","https://")) else decode_b64(source)

def save_input(source: Any, token: str, index: int):
    raw = source_bytes(source)
    try:
        with Image.open(BytesIO(raw)) as src:
            src.load()
            img = ImageOps.exif_transpose(src)
            w, h = img.size
            if w * h > MAX_INPUT_PIXELS:
                raise InputError("Input image has too many pixels")
            if img.mode not in ("RGB","RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
            name = f"rp_{token}_{index}.png"
            path = COMFY_INPUT_DIR / name
            img.save(path, "PNG")
            return name, (w,h), path
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Invalid image: {exc}") from exc

def collect_sources(data: dict[str,Any]) -> list[Any]:
    value = data.get("images")
    if value is None:
        sources = []
    elif isinstance(value, list):
        sources = value
    else:
        sources = [value]
    if not sources:
        for key in ("image","image_url","image_base64"):
            if data.get(key):
                sources = [data[key]]
                break
    if len(sources) > 10:
        raise InputError("At most 10 reference images are supported")
    return sources

def queue_workflow(workflow: dict[str,Any], client_id: str) -> str:
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    if r.status_code == 400:
        raise RuntimeError(f"ComfyUI workflow validation failed: {r.text}")
    r.raise_for_status()
    prompt_id = r.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return prompt_id")
    return prompt_id

def wait_result(prompt_id: str) -> dict[str,Any]:
    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        history = r.json()
        record = history.get(prompt_id)
        if record:
            for msg in (record.get("status") or {}).get("messages") or []:
                if isinstance(msg, list) and len(msg) > 1 and msg[0] == "execution_error":
                    payload = msg[1]
                    raise RuntimeError(f"ComfyUI execution failed: {payload.get('exception_message', payload) if isinstance(payload,dict) else payload}")
            if record.get("outputs"):
                return record
        time.sleep(0.25)
    raise TimeoutError(f"Generation exceeded JOB_TIMEOUT={JOB_TIMEOUT}s")

def fetch_outputs(record: dict[str,Any]) -> tuple[list[bytes], list[Path]]:
    blobs, paths = [], []
    for output in (record.get("outputs") or {}).values():
        if not isinstance(output, dict):
            continue
        for item in output.get("images", []):
            params = {"filename": item["filename"], "subfolder": item.get("subfolder",""), "type": item.get("type","output")}
            r = requests.get(f"{COMFY_URL}/view", params=params, timeout=60)
            r.raise_for_status()
            blobs.append(r.content)
            if params["type"] == "output":
                p = (COMFY_OUTPUT_DIR / params["subfolder"] / params["filename"]).resolve()
                try:
                    p.relative_to(COMFY_OUTPUT_DIR.resolve())
                    paths.append(p)
                except ValueError:
                    pass
    if not blobs:
        raise RuntimeError("ComfyUI completed without image output")
    return blobs, paths

def encode_output(blob: bytes, fmt: str, quality: int) -> str:
    fmt = fmt.lower().replace("jpg","jpeg")
    if fmt not in {"png","jpeg","webp"}:
        raise InputError("output_format must be png, jpeg or webp")
    with Image.open(BytesIO(blob)) as img:
        img.load()
        buf = BytesIO()
        if fmt == "jpeg":
            if img.mode not in ("RGB","L"):
                img = img.convert("RGB")
            img.save(buf, "JPEG", quality=max(1,min(100,quality)))
        elif fmt == "webp":
            img.save(buf, "WEBP", quality=max(1,min(100,quality)))
        else:
            img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def cleanup(paths: list[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception as exc:
            logger.warn(f"Cleanup failed for {p}: {exc}")

def handler(job: dict[str,Any]) -> dict[str,Any]:
    started = time.time()
    input_paths, output_paths = [], []
    try:
        data = job.get("input") or {}
        if not isinstance(data, dict):
            raise InputError("job.input must be an object")
        prompt = str(data.get("prompt","")).strip()
        if not prompt:
            raise InputError("Missing 'prompt' parameter")
        seed = int(data.get("seed", random.randint(0, 2**31 - 1)))
        steps = int(data.get("steps", 25))
        cfg = float(data.get("cfg_scale", 1.0))
        turbo = bool(data.get("turbo", False))
        loras = normalize_loras(data.get("loras", data.get("lora")))
        fmt = str(data.get("output_format","png"))
        quality = int(data.get("quality",95))
        sources = collect_sources(data)

        token = uuid.uuid4().hex[:12]
        image_names = []
        first_size = None
        for i, source in enumerate(sources,1):
            name, size, path = save_input(source, token, i)
            image_names.append(name)
            input_paths.append(path)
            first_size = first_size or size

        width, height = data.get("width"), data.get("height")
        if first_size and (width is None) != (height is None):
            sw, sh = first_size
            if width is None:
                width = round(int(height) * sw / sh)
            else:
                height = round(int(width) * sh / sw)

        workflow, actual_steps = build_workflow(
            prompt=prompt,
            negative_prompt=str(data.get("negative_prompt","") or ""),
            image_names=image_names,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg,
            seed=seed,
            loras=loras,
            turbo=turbo,
            reference_resolution=int(data.get("reference_resolution",1024)),
            model=MODEL,
            output_prefix=f"qwen21/{token}",
        )

        wait_for_comfy()
        logger.info(f"Qwen2.1 refs={len(image_names)} turbo={turbo} steps={actual_steps} loras={len(loras)} seed={seed}")
        prompt_id = queue_workflow(workflow, token)
        record = wait_result(prompt_id)
        blobs, output_paths = fetch_outputs(record)
        return {
            "images_base64": [encode_output(x, fmt, quality) for x in blobs],
            "time": round(time.time() - started, 2),
            "steps": actual_steps,
            "seed": seed,
            "turbo": turbo,
            "loras": loras,
        }
    except Exception as exc:
        logger.error(str(exc))
        return {"error": str(exc)}
    finally:
        cleanup(input_paths)
        cleanup(output_paths)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
