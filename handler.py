import runpod
from diffusers import FluxPipeline
import torch

# Load model and LoRA once on cold start
pipe = None

def load_model():
    global pipe
    if pipe is None:
        base_model = "black-forest-labs/FLUX.1-dev"
        lora_path = "/workspace/lora.safetensors"   # YOU WILL UPLOAD THIS FILE
        
        pipe = FluxPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16
        ).to("cuda")

        pipe.load_lora_weights(lora_path)

    return pipe

def generate_image(prompt: str):
    pipe = load_model()
    result = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=3.0
    )
    image = result.images[0]
    return image

def handler(event):
    prompt = event.get("input", {}).get("prompt", "gstyle apple clipart")

    image = generate_image(prompt)

    # Return base64 encoded image
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return { "image_base64": base64_image }

runpod.serverless.start({"handler": handler})
