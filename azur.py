import requests
import json
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Your endpoint and API key information
endpoint_url = "https://demo-for-diffusion-model--ghcuc.eastus.inference.ml.azure.com/score"  # Replace with your actual scoring URI
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your input image
input_image_path = "your_input_image.jpg"  # Replace with your image path

# Encode the image
base64_image = encode_image(input_image_path)

# Prepare headers with proper authentication
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'  # Note the 'Bearer ' prefix
}

# Request payload based on the interface shown
payload = {
    "prompt": "Studio Ghibli style astronaut in jungle, soft vibrant colors, detailed nature background",  # Your prompt
    "image": base64_image  # Base64 encoded image
    # You can add other parameters as needed based on your model configuration
}

# Make the request
print("Sending request to the endpoint...")
response = requests.post(endpoint_url, headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    # The response could be a JSON with base64 image or binary data
    try:
        # Try to parse as JSON
        result = response.json()
        
        # If the result contains a base64 image
        if 'image' in result:
            # Decode the base64 image
            image_data = base64.b64decode(result['image'])
            image = Image.open(BytesIO(image_data))
            
            # Save the image
            output_path = "generated_image.png"
            image.save(output_path)
            print(f"Image saved to {output_path}")
            
            # Display the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        else:
            print("Response received but no image found in the response")
            print(result)
    except json.JSONDecodeError:
        # If response is binary data (likely an image)
        with open("generated_image.png", "wb") as f:
            f.write(response.content)
        print("Image saved as generated_image.png")
        
        # Display the image
        image = Image.open(BytesIO(response.content))
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
else:
    print(f"Error: {response.status_code}")
    print(response.text)