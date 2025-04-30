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
url = "https://i.ibb.co/TMBkMFw8/reddy.jpg"
init_image = load_image(url)

# Prompt (anime-style, facial-preserving)
prompt = (
    "Sharp anime portrait of a smiling young man, Ghibli style, expressive eyes, well-defined facial features, "
    "clean lines, soft sunlight, vibrant natural background, artistic brushwork, smooth skin texture,"
    "high resolution, masterpiece quality, watercolor finish,"
)

# Strong negative prompt to avoid issues
negative_prompt = (
    "blurry face, distorted face, bad anatomy, multiple faces, extra limbs, double eyes, smudged skin, "
    "grainy, low resolution, low detail, watermark, glitch, broken face, soft focus."
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