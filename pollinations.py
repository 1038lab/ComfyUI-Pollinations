import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import requests
import tempfile
import time
from urllib.parse import quote, unquote

DEFAULT_IMAGE_MODELS = ["flux", "flux-pro", "flux-realism", "flux-anime", "flux-3d", "flux-cablyai", "turbo"]
DEFAULT_TEXT_MODELS = ["openai", "gpt-4", "gpt-3.5-turbo"]

MODELS_CACHE = {"models": [], "last_update": 0}
TEXT_MODELS_CACHE = {"models": [], "last_update": 0}

def get_available_models():
    """Get available image models from API with caching"""
    current_time = time.time()
    
    if current_time - MODELS_CACHE["last_update"] > 3600 or not MODELS_CACHE["models"]:
        try:
            response = requests.get("https://image.pollinations.ai/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if models_data and len(models_data) > 0:
                    MODELS_CACHE["models"] = models_data
                else:
                    MODELS_CACHE["models"] = DEFAULT_IMAGE_MODELS
                MODELS_CACHE["last_update"] = current_time
            else:
                MODELS_CACHE["models"] = DEFAULT_IMAGE_MODELS
        except Exception as e:
            print(f"Error fetching image models: {e}")
            MODELS_CACHE["models"] = DEFAULT_IMAGE_MODELS
    
    return MODELS_CACHE["models"]

def get_text_models():
    """Get available text models from API with caching"""
    current_time = time.time()
    
    if current_time - TEXT_MODELS_CACHE["last_update"] > 3600 or not TEXT_MODELS_CACHE["models"]:
        try:
            response = requests.get("https://text.pollinations.ai/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if models_data and len(models_data) > 0:
                    # Extract only model names from response
                    model_names = [model["name"] for model in models_data]
                    TEXT_MODELS_CACHE["models"] = model_names
                else:
                    TEXT_MODELS_CACHE["models"] = DEFAULT_TEXT_MODELS
                TEXT_MODELS_CACHE["last_update"] = current_time
            else:
                TEXT_MODELS_CACHE["models"] = DEFAULT_TEXT_MODELS
        except Exception as e:
            print(f"Error fetching text models: {e}")
            TEXT_MODELS_CACHE["models"] = DEFAULT_TEXT_MODELS
    
    return TEXT_MODELS_CACHE["models"]

class PollinationsImageGen:
    
    @classmethod
    def INPUT_TYPES(cls):
        models = get_available_models()
        default_model = "flux" if "flux" in models else models[0] if models else "flux"
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter a description of the image you want..."}),
                "model": (models, {"default": default_model}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enhance": ("BOOLEAN", {"default": True}),
                "nologo": ("BOOLEAN", {"default": True}),
                "private": ("BOOLEAN", {"default": True}),
                "safe": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "prompts")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/Pollinations"
    
    def generate(self, prompt, model, width, height, batch_size=1, negative_prompt="", seed=0, 
                 enhance=True, nologo=True, private=True, safe=False):
        """Generate multiple images"""
        images = []
        urls = []
        prompts = []
        
        for i in range(batch_size):
            current_seed = seed + i if seed != 0 else 0
            try:
                image, url, final_prompt = self._generate_single(
                    prompt, model, width, height, negative_prompt, 
                    current_seed, enhance, nologo, private, safe
                )
                images.append(image)
                urls.append(url)
                prompts.append(final_prompt)
            except Exception as e:
                print(f"Error generating image {i+1}: {e}")
                images.append(torch.zeros(1, 512, 512, 3))
                urls.append(f"Error: {str(e)}")
                prompts.append(prompt)
        
        return (images, urls, prompts)
    
    def _generate_single(self, prompt, model, width, height, negative_prompt="", seed=0, 
                        enhance=True, nologo=True, private=True, safe=False):
        """Generate a single image"""
        try:
            # Build base URL - using official API format from reference
            base_url = "https://image.pollinations.ai/prompt/"
            
            # Build full prompt
            full_prompt = prompt
            if negative_prompt:
                full_prompt = f"{prompt} ### {negative_prompt}"
                
            # URL encode the prompt
            encoded_prompt = quote(full_prompt)
            
            # Build parameters
            params = {}
            params["model"] = model
            params["width"] = width
            params["height"] = height
            
            if seed and seed != 0:
                params["seed"] = seed
            if nologo:
                params["nologo"] = "true"
            if private:
                params["private"] = "true"
            if enhance:
                params["enhance"] = "true"
            if safe:
                params["safe"] = "true"
            
            # Build complete URL
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}{encoded_prompt}?{param_str}"
            
            print(f"Generating image, URL: {url}")
            
            # Download image
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get the final prompt used (if enhanced)
            final_prompt = full_prompt  # Default to original prompt
            
            # Try to extract enhanced prompt from response URL
            try:
                image_url = response.url
                if "/prompt/" in image_url:
                    encoded_part = image_url.split("/prompt/")[1].split("?")[0]
                    extracted_prompt = unquote(encoded_part)
                    if extracted_prompt != full_prompt and enhance:
                        final_prompt = extracted_prompt
                        print(f"Enhanced prompt: {final_prompt}")
            except Exception as ee:
                print(f"Error extracting enhanced prompt: {ee}")
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            filename = f"pollinations_{int(time.time())}.png"
            image_path = os.path.join(temp_dir, filename)
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load image
            image = Image.open(image_path)
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
            
            return (image_tensor, url, final_prompt)
            
        except Exception as e:
            error_msg = f"Pollinations API error: {str(e)}"
            print(error_msg)
            # Return error message
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Ensure a new image is generated each time
        return time.time()

class PollinationsTextGen:
    @classmethod
    def INPUT_TYPES(cls):
        text_models = get_text_models()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter your text prompt..."}),
                "model": (text_models, {"default": "openai"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "private": ("BOOLEAN", {"default": True, "tooltip": "Keep the generation private"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "🧪AILab/Pollinations"
    
    def generate_text(self, prompt, model, seed, private=True):
        try:
            # Build URL with parameters
            params = {
                "model": model,
                "seed": seed,
                "private": str(private).lower()
            }
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(prompt)}?{param_str}"
            
            response = requests.get(url)
            if response.status_code == 200:
                return (response.text,)
            else:
                return (f"Error: {response.status_code}",)
        except Exception as e:
            return (f"Text generation failed: {str(e)}",)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "PollinationsImageGen": PollinationsImageGen,
    "PollinationsTextGen": PollinationsTextGen,
}

# UI display name
NODE_DISPLAY_NAME_MAPPINGS = {
    "PollinationsImageGen": "Pollinations Image Generation",
    "PollinationsTextGen": "Pollinations Text Generation",
} 