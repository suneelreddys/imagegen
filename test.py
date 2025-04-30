import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# Print PyTorch and CUDA info for debugging
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Load the pipeline
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Load an example local image
print("Loading local initial image...")
local_image_path = "/home/renotesuneel/your_image.png"  # <-- Change this to your local image path
init_image = Image.open(local_image_path).convert("RGB")
init_image = init_image.resize((768, 512))

# Generate an image
print("Generating new image...")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

# Create output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Save images
print(f"Saving images to {output_dir}...")
init_image.save(os.path.join(output_dir, "sd_input_image.png"))
image.save(os.path.join(output_dir, "sd_output_image.png"))

# Create a comparison grid
grid = Image.new('RGB', (init_image.width + image.width, max(init_image.height, image.height)))
grid.paste(init_image, (0, 0))
grid.paste(image, (init_image.width, 0))
grid.save(os.path.join(output_dir, "sd_comparison.png"))

print(f"Images saved to {os.path.abspath(output_dir)}")
