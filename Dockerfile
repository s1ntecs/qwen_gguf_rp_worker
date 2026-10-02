# Qwen-Image-2.1 GGUF RunPod Serverless worker.
# Base model components are baked into the image so cold workers do not spend
# paid GPU time downloading ~15 GB of weights.
FROM runpod/worker-comfyui:5.10.0-base@sha256:e9d18d3db15839ebb10c10c824109fc2956f31ef7bf21916220c008855b0d538

ARG COMFYUI_COMMIT=73c9bad4d21e7addbe1d13bc92eee0f1431b017d
ARG COMFYUI_GGUF_COMMIT=373048b8403a7820620065210a691263d4da0a61

ARG QWEN_GGUF_REPO=0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF
ARG QWEN_GGUF_REVISION=242be4d1f70cc26fa45773c452f81980ed656e2e
ARG QWEN_GGUF_FILE=qwen-image-2.1-Q4_K_M.gguf
ARG QWEN_GGUF_SHA256=833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8

ARG QWEN_COMPONENTS_REPO=Comfy-Org/Qwen-Image-2.1
ARG QWEN_COMPONENTS_REVISION=9a44dbdb47cefd046be9c0a13476192f34c8db8e
ARG QWEN_TEXT_ENCODER_PATH=text_encoders/qwen3vl_8b_int8_convrot.safetensors
ARG QWEN_TEXT_ENCODER_SHA256=8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f
ARG QWEN_VAE_PATH=vae/qwen_image_2.1_vae_bf16.safetensors
ARG QWEN_VAE_SHA256=bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9

ARG VIGGLE_REPO=Viggle/Qwen-Image-2.1-viggle-turbo
ARG VIGGLE_REVISION=bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13
ARG VIGGLE_LORA=Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors
ARG VIGGLE_LORA_SHA256=bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b

ENV QWEN_MODEL_FILE=${QWEN_GGUF_FILE} \
    QWEN_TEXT_ENCODER_FILE=qwen3vl_8b_int8_convrot.safetensors \
    QWEN_VAE_FILE=qwen_image_2.1_vae_bf16.safetensors \
    QWEN_TURBO_LORA_FILE=${VIGGLE_LORA} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# Native Qwen Image 2.1 nodes require a recent ComfyUI.  The leejet GGUF fork
# is pinned because it contains current Qwen Image 2.1 architecture handling.
RUN git -C /comfyui fetch --depth 1 origin ${COMFYUI_COMMIT} \
    && git -C /comfyui checkout --detach ${COMFYUI_COMMIT} \
    && rm -rf /comfyui/custom_nodes/ComfyUI-GGUF \
    && git clone https://github.com/leejet/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF \
    && git -C /comfyui/custom_nodes/ComfyUI-GGUF checkout --detach ${COMFYUI_GGUF_COMMIT}

COPY requirements.txt /app/requirements.txt
COPY scripts/patch_gguf_adapters.py /app/scripts/patch_gguf_adapters.py
RUN /opt/venv/bin/python /app/scripts/patch_gguf_adapters.py \
      /comfyui/custom_nodes/ComfyUI-GGUF/ops.py
RUN uv pip install --python /opt/venv/bin/python \
      -r /comfyui/requirements.txt \
      -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt \
      -r /app/requirements.txt

RUN mkdir -p \
      /comfyui/models/diffusion_models \
      /comfyui/models/text_encoders \
      /comfyui/models/vae \
      /comfyui/models/loras

COPY scripts/download_models.py /app/scripts/download_models.py
COPY lora_catalog.py /app/lora_catalog.py

# Each large asset gets its own Docker layer for better build-cache reuse.
RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_GGUF_REPO}" \
      --revision "${QWEN_GGUF_REVISION}" \
      --file "${QWEN_GGUF_FILE}" \
      --sha256 "${QWEN_GGUF_SHA256}" \
      --target "/comfyui/models/diffusion_models/${QWEN_GGUF_FILE}"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_COMPONENTS_REPO}" \
      --revision "${QWEN_COMPONENTS_REVISION}" \
      --file "${QWEN_TEXT_ENCODER_PATH}" \
      --sha256 "${QWEN_TEXT_ENCODER_SHA256}" \
      --target "/comfyui/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_COMPONENTS_REPO}" \
      --revision "${QWEN_COMPONENTS_REVISION}" \
      --file "${QWEN_VAE_PATH}" \
      --sha256 "${QWEN_VAE_SHA256}" \
      --target "/comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${VIGGLE_REPO}" \
      --revision "${VIGGLE_REVISION}" \
      --file "${VIGGLE_LORA}" \
      --sha256 "${VIGGLE_LORA_SHA256}" \
      --target "/comfyui/models/loras/${VIGGLE_LORA}" \
    && /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${VIGGLE_REPO}" \
      --revision "${VIGGLE_REVISION}" \
      --file "comfyui/viggle_turbo.py" \
      --target "/comfyui/custom_nodes/viggle_turbo.py"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --builtin-loras \
      --target /comfyui/models/loras

COPY custom_nodes/qwen_lora_manager.py /comfyui/custom_nodes/qwen_lora_manager.py
COPY lora_catalog.py /comfyui/lora_catalog.py
COPY handler.py workflow_builder.py extra_model_paths.yaml start.sh /app/

# Fail the build early if ComfyUI or custom nodes cannot import.
RUN chmod +x /app/start.sh \
    && cd /comfyui \
    && timeout 300 /opt/venv/bin/python main.py --quick-test-for-ci --cpu \
    && timeout 300 /opt/venv/bin/python -c \
      'import comfy.options; comfy.options.enable_args_parsing(); import asyncio, nodes; asyncio.run(nodes.init_extra_nodes()); required = {"UnetLoaderGGUFAdvanced", "TextEncodeQwenImage21", "QwenBuiltinLoraStack", "ViggleTurboLora", "ViggleTurboSigmas"}; missing = required - nodes.NODE_CLASS_MAPPINGS.keys(); assert not missing, f"Required ComfyUI nodes failed to import: {sorted(missing)}"' --cpu

WORKDIR /app
CMD ["/app/start.sh"]
