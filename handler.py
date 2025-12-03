import base64
from io import BytesIO

import runpod
import torch
from diffusers import AutoPipelineForText2Image

# Hugging Face info
BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_REPO_ID = "501miles/libericomm-lora"
LORA_WEIGHT_NAME = (
    "24f8d719-ad02-4163-ab5e-2aa2f4c66d4f-u2_627e2f15-25e9-4b67-9a89-4261811b9c73.safetensors"
)

pipe = None


def get_pipeline():
    """Load Flux + your LoRA once, then reuse."""
    global pipe
    if pipe is not None:
        return pipe

    pipe = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.load_lora_weights(
        LORA_REPO_ID,
        weight_name=LORA_WEIGHT_NAME,
    )

    return pipe


def generate(job):
    data = job.get("input", {}) or {}
    prompt = data.get("prompt", "a cute dog in a garden")

    pipe = get_pipeline()
    image = pipe(
        prompt,
        num_inference_steps=24,
        guidance_scale=3.5,
        height=768,
        width=768,
    ).images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image_base64": img_b64}


runpod.serverless.start({"handler": generate})
