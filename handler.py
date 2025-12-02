import base64
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline
import runpod

# Load Stable Diffusion 1.5 once when the worker starts
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)


def generate_image(prompt: str):
    """Generate an image and return it as base64 PNG."""
    image = pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def handler(event):
    """
    RunPod Serverless handler.
    Expected input:
    {
      "input": {
         "prompt": "a cute cat"
      }
    }
    """
    prompt = event["input"].get("prompt", "")
    result = generate_image(prompt)
    return {"image": result}


# Start the worker
runpod.serverless.start({"handler": handler})
