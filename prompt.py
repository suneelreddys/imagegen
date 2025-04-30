import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
from PIL import Image

# Initialize the pipeline
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",  # Corrected model repo ID
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# pipeline.enable_xformers_memory_efficient_attention()  # Uncomment if you have xFormers

# Load initial image
url = "https://i.ibb.co/B2RPzGg0/Screenshot-2025-04-29-173702.png"
init_image = load_image(url)

# Prompt (anime-style, facial-preserving)
prompt = (
    "Anime-style portrait of a smiling person, clean facial features, sharp eyes, natural lighting, realistic proportions, "
    "detailed clothing and accessories, vibrant watercolor shading, soft background, Ghibli-style, high quality artwork"
)


# Strong negative prompt to avoid issues
negative_prompt = (
    "blurry, distorted, extra limbs, multiple faces, bad anatomy, glitch, cropped face, watermark, low detail"
)


# Generate the image with improved parameters
result = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.45,               # lower = more like original image
    guidance_scale=9.0,          # higher = more like the prompt
    num_inference_steps=40       # more steps = better quality
).images[0]

# Make a comparison grid
grid_image = make_image_grid([init_image, result], rows=1, cols=2)
grid_image.save("output.png")

print("✅ Anime transformation saved as 'output.png'")