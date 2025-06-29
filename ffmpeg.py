# main.py

import os
import av
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

def capture_frame() -> Image.Image:
    """
    Capture one frame via AVFoundation/FFmpeg and return a PIL RGB image.
    """
    container = av.open(
        "1", # change this if you need to access the iphone camera
        format="avfoundation",
        options={
            "framerate": "30",
            "video_size": "1280x720",
            "pixel_format": "bgr0",   # one of uyvy422,yuyv422,nv12,0rgb,bgr0
        },
    )
    for frame in container.decode(video=0):
        rgb_arr = frame.to_ndarray(format="rgb24")
        return Image.fromarray(rgb_arr)
    raise RuntimeError("No frame decoded from camera")

def load_pipeline(device: str):
    """
    Load and return the Stable Diffusion Img2Img pipeline on `device`.
    """
    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="mps" else torch.float32,
    ).to(device)

def main():
    # 1) Device & pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[+] Using device: {device}")
    pipe = load_pipeline(device)
    print("[+] Pipeline loaded.")

    # 2) Capture & save the original
    orig = capture_frame()
    os.makedirs("outputs", exist_ok=True)
    orig.save("outputs/0_original.png")
    print("[+] Saved original → outputs/0_original.png")

    # 3) Set up prompt and which timesteps to inspect
    prompt = "make it look like an electric fan"
    num_steps = 50
    inspect_steps = {int(num_steps * frac) for frac in (0.25, 0.5, 0.75)}
    captured = {}

    # 4) Callback: note the signature!
    def capture_callback(pipeline, step_idx, timestep, callback_kwargs):
        # Extract the actual latents tensor:
        latents = callback_kwargs["latents"]
        if step_idx in inspect_steps:
            captured[step_idx] = latents.detach().clone()
        # Must return a dict mapping "latents" → tensor:
        return {"latents": latents}

    # 5) Run img2img with our callback_on_step_end
    result = pipe(
        prompt=prompt,
        image=orig.resize((512,512)),
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=num_steps,
        callback_on_step_end=capture_callback,
    )

    # 6) Decode & save each intermediate
    for step_idx, lat in sorted(captured.items()):
        with torch.no_grad():
            dec = pipe.vae.decode(lat).sample
        img = (dec / 2 + 0.5).clamp(0,1)
        arr = img.permute(0,2,3,1).cpu().numpy()[0]
        Image.fromarray((arr * 255).round().astype("uint8")).save(
            f"outputs/{step_idx:02d}_step.png"
        )
        print(f"[+] Saved step {step_idx} → outputs/{step_idx:02d}_step.png")

    # 7) Finally, save the stylized result
    result.images[0].save("outputs/5_final.png")
    print("[+] Saved final → outputs/5_final.png")

if __name__ == "__main__":
    main()
