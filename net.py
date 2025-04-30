import torch
import numpy as np
import cv2
from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import make_image_grid, load_image
from PIL import Image

# Load controlnet for better preservation of features
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", 
    torch_dtype=torch.float16
)

# Load SDXL pipeline with ControlNet
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()

# Load initial image
url = "https://i.ibb.co/B2RPzG0/Screenshot-2025-04-29-173702.png"
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

# Fix: Explicitly set the required parameters to avoid type error
# Get original image dimensions
width, height = init_image.size

# Generate the image with ControlNet for better detail preservation
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=canny_image,
    controlnet_conditioning_scale=0.75,  # Balanced control for structure
    strength=0.8,                # Higher for more stylization
    guidance_scale=8.5,          # Balanced for better results
    num_inference_steps=50,      # More steps for quality
    # Fix for the error: explicitly provide original_size, crop_coords, and aesthetic_score as integers
    original_size=(height, width),  # Order is (height, width) for SDXL
    target_size=(height, width),  
    crops_coords_top_left=(0, 0),
    aesthetic_score=6,  # An integer value between 0-10
    negative_aesthetic_score=2,
).images[0]

# Make a comparison grid with original and result
grid_image = make_image_grid([init_image, result], rows=1, cols=2)
grid_image.save("universal_ghibli_transformation.png")

print("✅ Universal Ghibli transformation saved as 'universal_ghibli_transformation.png'")