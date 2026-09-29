# Qwen Image 2.1 GGUF RunPod Serverless Worker

Production-oriented RunPod Serverless worker for **Qwen Image 2.1** using the GGUF transformer from **0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF**.

The public API is intentionally compact. Clients send prompt/settings/reference images; the worker builds and runs the ComfyUI workflow internally and returns base64 images in the same general shape as the existing Diffusers worker.

## Stack

- GGUF transformer: `0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF`
- Default quant: `qwen-image-2.1-Q4_K_M.gguf`
- Qwen3-VL INT8 text encoder: `Comfy-Org/Qwen-Image-2.1`
- Qwen Image 2.1 BF16 VAE
- ComfyUI with native Qwen Image 2.1 nodes
- `leejet/ComfyUI-GGUF` pinned to a Qwen-Image-2.1-capable commit
- Dynamic model-only LoRA chain from Network Volume
- Viggle Qwen Image 2.1 Turbo v0.2.1, exact 6-step sigma schedule
- T2I and image editing with up to 10 reference images
- URL, raw base64 and data-URI inputs
- PNG/JPEG/WebP base64 output

The 0xSojalSec repository contains GGUF quantizations of the original Qwen Image 2.1 transformer. It is not a separate fine-tune; encoder and VAE are required separately.

## Request API

### Text to image

```json
{
  "input": {
    "prompt": "A cinematic portrait, soft window light",
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "cfg_scale": 1.0,
    "seed": 42
  }
}
```

### Edit from URL

```json
{
  "input": {
    "prompt": "Change the jacket to dark green. Preserve identity, pose, framing and background.",
    "images": [
      "https://example.com/person.png"
    ],
    "steps": 25,
    "cfg_scale": 1.0,
    "seed": 42
  }
}
```

### Edit from base64

```json
{
  "input": {
    "prompt": "Replace the background with a snowy mountain landscape.",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAA..."
    ]
  }
}
```

Raw base64 without a data-URI prefix is also accepted.

### Multiple references

```json
{
  "input": {
    "prompt": "Put the person from <image1> in the outfit from <image2>.",
    "images": [
      "https://example.com/person.jpg",
      "https://example.com/outfit.jpg"
    ]
  }
}
```

Up to 10 reference images are supported.

### Single-image aliases

Instead of `images`, callers may use one of:

- `image`
- `image_url`
- `image_base64`

### Input fields

| Field | Default | Description |
|---|---:|---|
| `prompt` | required | Generation prompt or edit instruction |
| `negative_prompt` | `""` | Normal-mode negative prompt |
| `images` | `[]` | URL/base64/data URI/object; max 10 |
| `steps` | `25` | 1-100 normal mode; ignored by Turbo |
| `cfg_scale` | `1.0` | CFG for normal mode |
| `seed` | random | Integer seed |
| `width`, `height` | 1024x1024 for T2I | Rounded to multiples of 32 |
| `reference_resolution` | `1024` | Qwen conditioning resolution |
| `turbo` | `false` | Enable Viggle v0.2.1 6-step path |
| `loras` | `[]` | Additional LoRAs by filename/path |
| `output_format` | `png` | png/jpeg/webp |
| `quality` | `95` | JPEG/WebP quality |

## Response

```json
{
  "images_base64": ["iVBORw0KGgoAAA..."],
  "time": 14.82,
  "steps": 25,
  "seed": 42,
  "turbo": false,
  "loras": []
}
```

Error:

```json
{
  "error": "ComfyUI workflow validation failed: ..."
}
```

The key compatibility fields are `images_base64`, `time`, `steps`, and `seed`.

## LoRA support

LoRAs are applied with ComfyUI's `LoraLoaderModelOnly` after the GGUF model loader.

Place LoRA files on the Network Volume:

```text
/runpod-volume/models/loras/
  realism.safetensors
  identity/my_identity.safetensors
  camera/any_angle.safetensors
```

Request:

```json
{
  "input": {
    "prompt": "A realistic portrait",
    "loras": [
      {"name": "realism.safetensors", "strength": 0.8},
      {"name": "identity/my_identity.safetensors", "strength": 1.0}
    ]
  }
}
```

Short form:

```json
{
  "input": {
    "prompt": "A realistic portrait",
    "loras": ["realism.safetensors"]
  }
}
```

In Turbo mode the chain is:

```text
GGUF base -> Viggle Turbo LoRA -> custom LoRA 1 -> custom LoRA 2 -> sampler
```

Compatibility still depends on how a LoRA was trained. A LoRA trained for a different Qwen generation/edit family may fail to match keys or produce poor results.

The worker intentionally does not download arbitrary per-request LoRA URLs. Put them on the Network Volume for predictable cold starts and safer operation.

## Turbo mode

```json
{
  "input": {
    "prompt": "Turn this room into a modern Scandinavian interior",
    "images": ["https://example.com/room.jpg"],
    "turbo": true,
    "seed": 123
  }
}
```

Turbo uses:

`Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`

plus the six-step sigma sequence shipped for Viggle Turbo.

The response reports `"steps": 6`.

For difficult identity-preserving and multi-reference edits, normal 25-40 step mode can be more faithful.

## Why ComfyUI is inside the worker

This GGUF image transformer is loaded through ComfyUI-GGUF. The RunPod handler exposes a normal JSON API while ComfyUI stays private on `127.0.0.1:8188`. API clients never need to provide a ComfyUI workflow.

## Build locally

```bash
docker build --platform linux/amd64 -t yourname/qwen21-gguf-runpod:latest .
docker push yourname/qwen21-gguf-runpod:latest
```

The image bakes the base transformer, encoder, VAE and Turbo LoRA so cold workers do not download model weights during paid GPU startup.

To build another GGUF quant:

```bash
docker build --platform linux/amd64 \
  --build-arg QWEN_GGUF_FILE=qwen-image-2.1-Q5_K_M.gguf \
  --build-arg QWEN_GGUF_SHA256=88ce8e90e5b959cce5e248f697d7f6c9c7ca5696c1eac64a10dadb041dd7fd07 \
  -t yourname/qwen21-gguf-runpod:q5 .
```

Available 0xSojalSec variants currently include Q4_0, Q4_K_M, Q5_K_M, Q6_K and Q8_0. Q4_K_M is the model repository's recommended balance of size and quality.

## Pinned model/runtime revisions

The Docker build is intentionally reproducible and verifies the large model files before they are copied into the final image.

| Component | Revision / checksum |
|---|---|
| `0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF` | `242be4d1f70cc26fa45773c452f81980ed656e2e` |
| Q4_K_M GGUF SHA256 | `833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8` |
| ComfyUI | `73c9bad4d21e7addbe1d13bc92eee0f1431b017d` |
| leejet/ComfyUI-GGUF | `373048b8403a7820620065210a691263d4da0a61` |
| Comfy-Org Qwen components | `9a44dbdb47cefd046be9c0a13476192f34c8db8e` |
| Viggle Turbo | `bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13` |

If you override the GGUF file at build time, also pass the matching `QWEN_GGUF_SHA256`.

## GitHub Container Registry

The repository includes `.github/workflows/build-ghcr.yml`.

It builds on manual dispatch and on tags matching `v*`.

Image:

```text
ghcr.io/s1ntecs/qwen_gguf_rp_worker:latest
```

If RunPod should pull it without credentials, make the GHCR package public.

## RunPod Serverless setup

Recommended starting point:

- NVIDIA GPU with 24 GB VRAM or more
- container disk 35-45 GB
- min workers 0 during testing
- max workers according to traffic
- optional Network Volume for custom LoRAs

No public ComfyUI port is required.

Example:

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cinematic portrait, soft window light",
      "width": 1024,
      "height": 1024,
      "turbo": true,
      "seed": 42
    }
  }'
```

For large base64 requests use `/run` plus `/status/<job-id>` instead of `/runsync`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_MODEL_FILE` | `qwen-image-2.1-Q4_K_M.gguf` | GGUF transformer |
| `QWEN_TEXT_ENCODER_FILE` | `qwen3vl_8b_int8_convrot.safetensors` | Encoder |
| `QWEN_VAE_FILE` | `qwen_image_2.1_vae_bf16.safetensors` | VAE |
| `QWEN_TURBO_LORA_FILE` | Viggle v0.2.1 r128 | Turbo adapter |
| `JOB_TIMEOUT` | `900` | Generation timeout |
| `COMFY_READY_TIMEOUT` | `180` | Startup timeout |
| `DOWNLOAD_TIMEOUT` | `45` | Input URL timeout |
| `MAX_INPUT_BYTES` | `31457280` | Per-input byte limit |
| `MAX_INPUT_PIXELS` | `40000000` | Decode-bomb protection |
| `ALLOW_PRIVATE_IMAGE_URLS` | `false` | Keep false on public endpoints |
| `COMFY_PERFORMANCE_ARGS` | `--fast fp16_accumulation` | ComfyUI runtime flags |

## Security notes

- Remote image URLs are checked against private, loopback, link-local, multicast and reserved IP ranges to reduce SSRF risk.
- Redirect targets are validated again.
- Input size and decoded pixel count are bounded.
- LoRA paths reject absolute paths and `..`.
- ComfyUI listens only on loopback inside the container.

## Tests

Workflow construction tests do not need a GPU:

```bash
python -m unittest discover -s tests -v
```

A full integration test requires a CUDA GPU and the built container.

## License

The Qwen model is distributed under the Qwen Research License. Review the current model license before commercial deployment. This worker does not modify or expand the model license.
