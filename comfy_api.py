# comfy_api.py
# This is the ComfyUI API script (formerly NEW-IMAGE-GEN.py)

import websocket  # Make sure this is websocket-client, not websockets
import uuid
import json
import urllib.request
import urllib.parse
import random
import os

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

# Get the directory of the current script to build absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))

def queue_prompt(workflow):
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(workflow_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, workflow_id)) as response:
        return json.loads(response.read())

def get_images(ws, workflow):
    workflow_id = queue_prompt(workflow)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == workflow_id:
                    break #Execution is done

    history = get_history(workflow_id)[workflow_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


def generate_image_flux(prompt_text, output_path, seed=None, width=1024, height=1024):
    """
    Generate a single image using Flux model via ComfyUI API
    
    Args:
        prompt_text: The text prompt for image generation
        output_path: Path where the image should be saved
        seed: Random seed (optional)
        width: Image width
        height: Image height
    """
    # Load the workflow using an absolute path
    workflow_path = os.path.join(script_dir, "flux-kontext.json")
    with open(workflow_path, "r") as f:
        workflow_text = f.read()
    
    workflow = json.loads(workflow_text)
    
    # Set the prompt
    workflow["20"]["inputs"]["text"] = prompt_text
    
    # Set the seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    workflow["29"]["inputs"]["seed"] = seed
    
    # Set dimensions
    workflow["27"]["inputs"]["height"] = height
    workflow["27"]["inputs"]["width"] = width
    
    # Connect to websocket using create_connection for better compatibility
    ws = websocket.create_connection("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    try:
        # Generate images
        images = get_images(ws, workflow)
        
        # Save the first generated image to the specified path
        for node_id in images:
            if images[node_id]:  # If there are images from this node
                image_data = images[node_id][0]  # Get the first image
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path, "PNG")
                return True
        
        return False
    finally:
        ws.close()


def generate_image_sd(prompt_text, output_path, negative_prompt="text, watermark", seed=None, width=512, height=512):
    """
    Generate a single image using Stable Diffusion model via ComfyUI API
    
    Args:
        prompt_text: The text prompt for image generation
        output_path: Path where the image should be saved
        negative_prompt: Negative prompt text (what to avoid)
        seed: Random seed (optional)
        width: Image width
        height: Image height
    """
    # Load the SD workflow using an absolute path
    workflow_path = os.path.join(script_dir, "SD.json")
    with open(workflow_path, "r") as f:
        workflow_text = f.read()
    
    workflow = json.loads(workflow_text)
    
    # Set the positive prompt (node 6)
    workflow["6"]["inputs"]["text"] = prompt_text
    
    # Set the negative prompt (node 7)
    workflow["7"]["inputs"]["text"] = negative_prompt
    
    # Set the seed (node 3 - KSampler)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    workflow["3"]["inputs"]["seed"] = seed
    
    # Set dimensions (node 5 - Empty Latent Image)
    workflow["5"]["inputs"]["height"] = height
    workflow["5"]["inputs"]["width"] = width
    
    # Connect to websocket using create_connection for better compatibility
    ws = websocket.create_connection("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    try:
        # Generate images
        images = get_images(ws, workflow)
        
        # Save the first generated image to the specified path
        for node_id in images:
            if images[node_id]:  # If there are images from this node
                image_data = images[node_id][0]  # Get the first image
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path, "PNG")
                return True
        
        return False
    finally:
        ws.close()


if __name__ == "__main__":
    # Test the functions
    test_prompt = "A beautiful landscape with mountains and a lake"
    
    # Test Flux
    print("Testing Flux model...")
    flux_output = "test_flux_image.png"
    success = generate_image_flux(test_prompt, flux_output)
    if success:
        print(f"Flux image saved to {flux_output}")
    else:
        print("Failed to generate Flux image")
    
    # Test SD
    print("\nTesting SD model...")
    sd_output = "test_sd_image.png"
    success = generate_image_sd(test_prompt, sd_output)
    if success:
        print(f"SD image saved to {sd_output}")
    else:
        print("Failed to generate SD image")