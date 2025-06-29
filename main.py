import cv2, torch, os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np

# 1) Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="mps" else torch.float32
).to(device)

# 2) Capture from webcam

def brighten_frame(frame, alpha=2.3, beta=200):
    # alpha: contrast multiplier (>1 brightens)
    # beta:  brightness offset (0–100)
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def gamma_correction(frame, gamma=1.4):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = table.astype("uint8")
    return cv2.LUT(frame, table)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to grab frame")
frame = brighten_frame(frame, alpha=1.5, beta=75)
# frame = gamma_correction(frame, gamma=1.8)
orig = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512,512))

# 3) Choose your prompt and the timesteps you want to inspect
prompt = "make it look like an electric fan"
num_steps = 50
# pick three roughly-evenly spaced steps before the end
inspect_steps = sorted({int(num_steps * f) for f in (0.25, 0.5, 0.75)})

# 4) Prepare a place to stash latents
captured = {}

# 5) Define our callback
def capture_callback(step_idx, timestep, latents):
    # step_idx runs 0..num_steps-1
    if step_idx in inspect_steps:
        # clone so we don't get overwritten
        captured[step_idx] = latents.detach().clone()

# 6) Run the pipeline with the callback
result = pipe(
    prompt=prompt,
    image=orig,
    strength=0.75,
    guidance_scale=7.5,
    num_inference_steps=num_steps,
    callback=capture_callback,
    callback_steps=1,            # ask to call us every single step
)

# make sure output folder exists
os.makedirs("inspection", exist_ok=True)

# 7) Save the original and final
orig.save("inspection/0_original.png")
final = result.images[0]
final.save("inspection/5_final.png")

# 8) Decode & save each captured latent
for step_idx, lat in captured.items():
    with torch.no_grad():
        # decode the latent back to image-space
        img_tensor = pipe.vae.decode(lat).sample
    # normalize & convert to PIL
    img = (img_tensor / 2 + 0.5).clamp(0,1)
    img = img.permute(0,2,3,1).cpu().numpy()[0]
    stage = Image.fromarray((img * 255).round().astype("uint8"))
    stage.save(f"inspection/{step_idx:02d}_step.png")

print("Saved all stages into ./inspection/")   
