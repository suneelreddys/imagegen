import torch
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import make_image_grid, load_image
from PIL import Image

# Load ControlNet for canny edge conditioning
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# Load SD 1.5 pipeline with ControlNet
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None  # Optional: disable safety checker if needed
)

# Move to GPU and enable memory optimizations
pipeline.enable_model_cpu_offload()
# pipeline.enable_xformers_memory_efficient_attention()  # Uncomment if you have xformers installed

# Load initial image
url = "https://i.ibb.co/B2RPzGg0/Screenshot-2025-04-29-173702.png"
init_image = load_image(url)

# Convert PIL image to numpy array for Canny edge detection
init_image_np = np.array(init_image)

# Apply Canny edge detection
low_threshold = 100
high_threshold = 200
canny_image = cv2.Canny(init_image_np, low_threshold, high_threshold)

# Convert back to RGB (3 channels) as expected by the model
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)

# Universal Ghibli style prompt that works for any image
prompt = (
    "Studio Ghibli art style, hand-painted by Hayao Miyazaki, traditional animation, "
    "soft watercolor painting, gentle brushstrokes, pastel color palette, "
    "detailed backgrounds, simple facial features, warm atmospheric lighting, "
    "artistic and whimsical, distinctive Ghibli aesthetic, nostalgic feeling, "
    "2D animation style, masterpiece quality, beautifully illustrated"
)

# Enhanced negative prompt
negative_prompt = (
    "3d, cgi, render, photorealistic, photography, realistic, hyperrealistic, "
    "low quality, pixelated, grainy, blurry, noise, text, watermark, signature, "
    "distorted anatomy, deformed, extra limbs, mutated, ugly, oversaturated"
)

# Generate the image with ControlNet
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=canny_image,
    controlnet_conditioning_scale=1.0,  # Higher control strength for better structure preservation
    strength=0.75,                # Balance between original image and new style
    guidance_scale=7.5,           # Prompt adherence strength
    num_inference_steps=40        # Number of denoising steps
).images[0]

# Make a comparison grid with original and result
grid_image = make_image_grid([init_image, result], rows=1, cols=2)
grid_image.save("ghibli_transformation_sd15.png")

print("✅ Ghibli transformation saved as 'ghibli_transformation_sd15.png'")