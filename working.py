import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
from PIL import Image
import matplotlib.pyplot as plt
import os
from google.colab import files  # Only needed if running in Google Colab

# Function to load a local image
def load_local_image(image_path):
    """Load an image from a local path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return Image.open(image_path)

# Check if running in Colab to use file upload functionality
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# Load the pipeline
print("Loading the SDXL pipeline...")
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# Remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# Get the input image
if IN_COLAB:
    print("Please upload an image...")
    uploaded = files.upload()
    
    # Get the filename of the first uploaded file
    image_path = list(uploaded.keys())[0]
    print(f"Using uploaded image: {image_path}")
else:
    # If not in Colab, specify a local path
    image_path = input("/content/WhatsApp Image 2025-02-24 at 23.16.53 (1).jpeg")

# Load the image
try:
    init_image = load_local_image(image_path)
    print(f"Image loaded successfully: {init_image.size}")
    
    # Resize the image if needed (SDXL works well with images divisible by 8)
    width, height = init_image.size
    if width % 8 != 0 or height % 8 != 0:
        # Calculate new dimensions that are divisible by 8
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        init_image = init_image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Image resized to {new_width}x{new_height} to ensure dimensions are divisible by 8")
    
    # Display the input image
    plt.figure(figsize=(8, 8))
    plt.imshow(init_image)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()
    
    # Define your prompt
    prompt = """Studio Ghibli style astronaut in jungle, soft vibrant colors, detailed nature background, 
    expressive character design, whimsical, hand-drawn animation look, watercolor effects"""
    
    print(f"Processing with prompt: '{prompt}'")
    print("Generating image (this may take a few minutes)...")
    
    # Set strength parameter (1.0 for complete transformation, lower values preserve more of original)
    strength = 0.75  # You can adjust this value between 0.0 and 1.0
    
    # Pass prompt and image to pipeline
    image = pipeline(
        prompt=prompt, 
        image=init_image, 
        strength=strength,
        num_inference_steps=30,  # You can adjust for faster/better quality
        guidance_scale=7.5       # You can adjust for more/less prompt adherence
    ).images[0]
    
    # Create a grid of the input and output images
    grid = make_image_grid([init_image, image], rows=1, cols=2)
    
    # Display the images directly in the notebook
    plt.figure(figsize=(16, 8))
    plt.imshow(grid)
    plt.title("Before and After")
    plt.axis('off')
    plt.show()
    
    # Save the individual output image
    output_filename = "output_" + os.path.basename(image_path)
    image.save(output_filename)
    
    # Save the grid of input and output images
    grid_filename = "comparison_" + os.path.basename(image_path)
    grid.save(grid_filename)
    
    print(f"Images saved as '{output_filename}' and '{grid_filename}'")
    
    # If in Colab, also offer to download the files
    if IN_COLAB:
        files.download(output_filename)
        files.download(grid_filename)
        print("Downloads initiated!")

except Exception as e:
    print(f"Error: {e}")