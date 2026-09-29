FROM runpod/worker-comfyui:5.10.0-base@sha256:e9d18d3db15839ebb10c10c824109fc2956f31ef7bf21916220c008855b0d538

ARG COMFYUI_COMMIT=73c9bad4d21e7addbe1d13bc92eee0f1431b017d
ARG COMFYUI_GGUF_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a

ARG QWEN_GGUF_REPO=0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF
ARG QWEN_GGUF_REVISION=242be4d
ARG QWEN_GGUF_FILE=qwen-image-2.1-Q4_K_M.gguf

ARG QWEN_COMPONENTS_REPO=Comfy-Org/Qwen-Image-2.1
ARG QWEN_COMPONENTS_REVISION=9a44dbdb47cefd046be9c0a13476192f34c8db8e
ARG QWEN_TEXT_ENCODER_PATH=text_encoders/qwen3vl_8b_int8_convrot.safetensors
ARG QWEN_VAE_PATH=vae/qwen_image_2.1_vae_bf16.safetensors

ARG VIGGLE_REPO=Viggle/Qwen-Image-2.1-viggle-turbo
ARG VIGGLE_REVISION=bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13
ARG VIGGLE_LORA=Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors

ENV QWEN_MODEL_FILE=${QWEN_GGUF_FILE} \
    QWEN_TEXT_ENCODER_FILE=qwen3vl_8b_int8_convrot.safetensors \
    QWEN_VAE_FILE=qwen_image_2.1_vae_bf16.safetensors \
    QWEN_TURBO_LORA_FILE=${VIGGLE_LORA} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

RUN git -C /comfyui fetch --depth 1 origin ${COMFYUI_COMMIT} \
    && git -C /comfyui checkout --detach ${COMFYUI_COMMIT} \
    && rm -rf /comfyui/custom_nodes/ComfyUI-GGUF \
    && git clone https://github.com/city96/ComfyUI-GGUF.git /comfyui/custom_nodes/ComfyUI-GGUF \
    && git -C /comfyui/custom_nodes/ComfyUI-GGUF checkout --detach ${COMFYUI_GGUF_COMMIT}

COPY requirements.txt /app/requirements.txt
RUN uv pip install --python /opt/venv/bin/python \
      -r /comfyui/requirements.txt \
      -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt \
      -r /app/requirements.txt

RUN mkdir -p /comfyui/models/diffusion_models /comfyui/models/text_encoders /comfyui/models/vae /comfyui/models/loras
COPY scripts/download_models.py /app/scripts/download_models.py

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_GGUF_REPO}" --revision "${QWEN_GGUF_REVISION}" \
      --file "${QWEN_GGUF_FILE}" \
      --target "/comfyui/models/diffusion_models/${QWEN_GGUF_FILE}"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_COMPONENTS_REPO}" --revision "${QWEN_COMPONENTS_REVISION}" \
      --file "${QWEN_TEXT_ENCODER_PATH}" \
      --target "/comfyui/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${QWEN_COMPONENTS_REPO}" --revision "${QWEN_COMPONENTS_REVISION}" \
      --file "${QWEN_VAE_PATH}" \
      --target "/comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors"

RUN /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${VIGGLE_REPO}" --revision "${VIGGLE_REVISION}" \
      --file "${VIGGLE_LORA}" \
      --target "/comfyui/models/loras/${VIGGLE_LORA}" \
    && /opt/venv/bin/python /app/scripts/download_models.py \
      --repo "${VIGGLE_REPO}" --revision "${VIGGLE_REVISION}" \
      --file "comfyui/viggle_turbo.py" \
      --target "/comfyui/custom_nodes/viggle_turbo.py"

COPY handler.py workflow_builder.py extra_model_paths.yaml start.sh /app/
RUN chmod +x /app/start.sh \
    && cd /comfyui \
    && timeout 300 /opt/venv/bin/python main.py --quick-test-for-ci --cpu

WORKDIR /app
CMD ["/app/start.sh"]
