import os, time, av, torch, requests
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

FLASH_HOST = "192.168.1.247"
FLASH_PORT = 8394

def trigger_iphone_flash(host, port, timeout=1.0):
    url = f"http://{host}:{port}/flash"
    try:
        requests.get(url, timeout=timeout)
        time.sleep(0.15)
    except Exception as e:
        print(f"[!] Could not trigger flash: {e}")

def capture_frame() -> Image.Image:
    container = av.open(
        "1", format="avfoundation",
        options={"framerate":"30","video_size":"1280x720","pixel_format":"bgr0"},
    )
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")
        return Image.fromarray(rgb)
    raise RuntimeError("No frame decoded")

def load_pipeline(device: str):
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="mps" else torch.float32,
    ).to(device)

def main():
    # Trigger the flash once, with correct host/IP
    print("[*] Triggering iPhone flash…")
    trigger_iphone_flash(FLASH_HOST, FLASH_PORT)
    print("[+] Flash triggered, capturing frame…")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[+] Using device: {device}")
    pipe = load_pipeline(device)
    print("[+] Pipeline loaded.")

    orig = capture_frame()
    os.makedirs("outputs", exist_ok=True)
    orig.save("outputs/0_original.png")
    print("[+] Saved original → outputs/0_original.png")

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
        image=orig.resize((512,512)),
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=num_steps,
        callback_on_step_end=capture_callback,
    )

    for idx, lat in sorted(captured.items()):
        with torch.no_grad():
            dec = pipe.vae.decode(lat).sample
        img = (dec / 2 + 0.5).clamp(0,1)
        arr = img.permute(0,2,3,1).cpu().numpy()[0]
        Image.fromarray((arr*255).astype("uint8")).save(f"outputs/{idx:02d}_step.png")
        print(f"[+] Saved step {idx} → outputs/{idx:02d}_step.png")

    result.images[0].save("outputs/5_final.png")
    print("[+] Saved final → outputs/5_final.png")

if __name__ == "__main__":
    main()