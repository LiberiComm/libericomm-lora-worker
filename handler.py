import runpod
from diffusers import StableDiffusionPipeline
import torch

# Load model once when the serverless container starts
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")


def generate_image(job):
    """
    job["input"] = { "prompt": "a red apple" }
    """
    prompt = job["input"].get("prompt", "a simple test prompt")

    image = pipe(prompt).images[0]

    # Return the image encoded as base64
    import base64
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image_base64": encoded}


runpod.serverless.start({"handler": generate_image})
