import torch
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import make_image_grid, load_image
from PIL import Image

# Load TWO ControlNet models - one for canny edges (structure) and one for faces
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

# Second ControlNet specifically for face preservation
controlnet_face = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", # Better for preserving poses and facial structure
    torch_dtype=torch.float16
)

# Load SD pipeline with both ControlNets
pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_face],
    torch_dtype=torch.float16,
    safety_checker=None
)

# Move to GPU and enable memory optimizations
pipeline.enable_model_cpu_offload()

# Load initial image
url = "https://i.ibb.co/B2RPzGg0/Screenshot-2025-04-29-173702.png"
init_image = load_image(url)

# Create canny edge map for structure preservation
init_image_np = np.array(init_image)
canny_image = cv2.Canny(init_image_np, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)

# Use the original image as the second control image (for face preservation)
# The face detection will happen automatically in the model
face_image = init_image

# Improved prompt with more emphasis on recognizable features
prompt = (
    "Ghibli style, detailed anime portrait, clear face, watercolor, Miyazaki art, smooth details"
)

# Enhanced negative prompt
negative_prompt = (
    "blurry, distorted, low quality, deformed face, bad anatomy"
)

# Generate the image with dual ControlNets
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=[canny_image, face_image],
    controlnet_conditioning_scale=[0.7, 0.7],  # Balance structure and face preservation
    guidance_scale=8.0,          # Stronger prompt adherence
    strength=0.5,               # Lower strength to preserve more of the original image 
    num_inference_steps=50,      # More steps for better quality
    seed=42                      # Fixed seed for reproducibility
).images[0]

# Make a comparison grid
grid_image = make_image_grid([init_image, result], rows=1, cols=2)
grid_image.save("improved_ghibli_transformation.png")

print("✅ Improved Ghibli transformation saved as 'improved_ghibli_transformation.png'")