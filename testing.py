import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import os

# Load the pipeline
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipeline.enable_model_cpu_offload()



# Prepare input image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

# Define the prompt
prompt = "Turn this image into a beautiful, detailed Studio Ghibli-style painting. Use soft pastel colors, cinematic lighting, delicate hand-drawn lines, a whimsical and dreamlike atmosphere, highly detailed backgrounds, classic Ghibli anime style."

# Generate the output image
output = pipeline(prompt=prompt, image=init_image)
generated_image = output.images[0]

# Create output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Save input and output images
print(f"Saving images to {output_dir}...")
init_image.save(os.path.join(output_dir, "kandinsky_input_image.png"))
generated_image.save(os.path.join(output_dir, "kandinsky_output_image.png"))

# Create and save a comparison grid
comparison_grid = make_image_grid([init_image, generated_image], rows=1, cols=2)
comparison_grid.save(os.path.join(output_dir, "kandinsky_comparison_grid.png"))

print(f"Images saved successfully at {os.path.abspath(output_dir)}")
