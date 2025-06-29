import os
import time

import av
import torch
import requests
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

FLASH_HOST = "192.168.1.247"
FLASH_PORT = 8395  # <-- make sure this matches the port your Swift server is actually listening on

def trigger_iphone_flash(host: str, port: int, timeout: float = 1.0):
    url = f"http://{host}:{port}/flash"
    try:
        resp = requests.get(url, timeout=timeout)
        print(f"[*] Flash request → {url} returned {resp.status_code}")
        # give the torch ~50–100 ms to come up
        time.sleep(0.05)
    except Exception as e:
        print(f"[!] Could not trigger flash: {e}")

def capture_frame(host: str, port: int, timeout: float = 1.0) -> Image.Image:
    """
    Opens the Continuity camera, fires the flash, then blocks
    until it actually decodes one torch-lit frame.
    """
    container = av.open(
        "0", format="avfoundation",
        options={
            "framerate": "30",
            "video_size": "1280x720",
            "pixel_format": "bgr0",
        },
    )
    video_stream = container.streams.video[0]

    # 1) Trigger the flash
    print("[*] Triggering flash…")
    trigger_iphone_flash(host, port, timeout)

    # 2) Now poll packets until we decode one frame
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            rgb = frame.to_ndarray(format="rgb24")
            img = Image.fromarray(rgb)
            container.close()
            return img

    # If we ever drop out of the loop, it's an error
    container.close()
    raise RuntimeError("Failed to decode any frame after flash")

def load_pipeline(device: str):
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    )
    return pipe.to(device)

def main():
    # 1) Pick device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # 2) Load model
    pipe = load_pipeline(device)
    print("[+] Pipeline loaded.")

    # 3) Capture the flash-lit frame
    orig = capture_frame(FLASH_HOST, FLASH_PORT)
    os.makedirs("outputs", exist_ok=True)
    orig.save("outputs/0_original.png")
    print("[+] Saved original → outputs/0_original.png")

    # 4) Set up diffusion
    prompt = "make it look like an electric fan"
    num_steps = 50
    inspect_steps = {int(num_steps * f) for f in (0.25, 0.5, 0.75)}
    captured = {}

    def capture_callback(pipeline, step_idx, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        if step_idx in inspect_steps:
            captured[step_idx] = lat.detach().clone()
        return {"latents": lat}

    # 5) Run the pipeline
    result = pipe(
        prompt=prompt,
        image=orig.resize((512, 512)),
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=num_steps,
        callback_on_step_end=capture_callback,
    )

    # 6) Decode & save intermediates
    for idx, lat in sorted(captured.items()):
        with torch.no_grad():
            dec = pipe.vae.decode(lat).sample
        img = (dec / 2 + 0.5).clamp(0, 1)
        arr = img.permute(0, 2, 3, 1).cpu().numpy()[0]
        Image.fromarray((arr * 255).astype("uint8")).save(f"outputs/{idx:02d}_step.png")
        print(f"[+] Saved step {idx} → outputs/{idx:02d}_step.png")

    # 7) Save final
    result.images[0].save("outputs/5_final.png")
    print("[+] Saved final → outputs/5_final.png")

if __name__ == "__main__":
    main()
