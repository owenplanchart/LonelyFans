import os
import io
import time

import torch
import requests
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

FLASH_HOST = "192.168.1.247"
FLASH_PORT = 8395

def fetch_flash_capture(host: str, port: int, timeout: float = 5.0) -> Image.Image:
    """
    Hits the Swift server's /capture endpoint, receives a flash-lit JPEG,
    and returns it as a PIL.Image.
    """
    url = f"http://{host}:{port}/capture"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def load_pipeline(device: str):
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    )
    return pipe.to(device)

def main():
    # 1) Pick your accelerator
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # 2) Load Stable Diffusion Img2Img
    pipe = load_pipeline(device)
    print("[+] Pipeline loaded.")

    # 3) Fetch the flash-lit capture from your Swift app
    print("[*] Fetching flash-lit image from http://"
          f"{FLASH_HOST}:{FLASH_PORT}/capture …")
    orig = fetch_flash_capture(FLASH_HOST, FLASH_PORT)
    os.makedirs("outputs", exist_ok=True)
    orig.save("outputs/0_original.png")
    print("[+] Saved original → outputs/0_original.png")

    # 4) Run your diffusion pipeline as before
    prompt = "make it look like an electric fan"
    num_steps = 50
    inspect_steps = {int(num_steps * f) for f in (0.25, 0.5, 0.75)}
    captured = {}

    def capture_callback(pipeline, step_idx, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        if step_idx in inspect_steps:
            captured[step_idx] = lat.detach().clone()
        return {"latents": lat}

    result = pipe(
        prompt=prompt,
        image=orig.resize((512, 512)),
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=num_steps,
        callback_on_step_end=capture_callback,
    )

    # 5) Save intermediates
    for idx, lat in sorted(captured.items()):
        with torch.no_grad():
            dec = pipe.vae.decode(lat).sample
        img = (dec / 2 + 0.5).clamp(0, 1)
        arr = img.permute(0, 2, 3, 1).cpu().numpy()[0]
        Image.fromarray((arr * 255).astype("uint8")).save(f"outputs/{idx:02d}_step.png")
        print(f"[+] Saved step {idx} → outputs/{idx:02d}_step.png")

    # 6) Save final
    result.images[0].save("outputs/5_final.png")
    print("[+] Saved final → outputs/5_final.png")

if __name__ == "__main__":
    main()
