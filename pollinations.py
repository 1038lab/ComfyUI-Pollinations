import os
import sys
import json
import numpy as np
import torch
import torchaudio
from PIL import Image
import requests
import tempfile
import time
from urllib.parse import quote, unquote
from pydub import AudioSegment
import io
import folder_paths

# Updated based on https://image.pollinations.ai/models
DEFAULT_IMAGE_MODELS = ["flux", "turbo"]
# Top models from https://text.pollinations.ai/models (first few models)
DEFAULT_TEXT_MODELS = ["openai", "openai-fast", "openai-large", "qwen-coder", "llama", "mistral"]

MODELS_CACHE = {"models": [], "last_update": 0}
TEXT_MODELS_CACHE = {"model_info": [], "last_update": 0}

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
    
    if current_time - TEXT_MODELS_CACHE["last_update"] > 3600 or not TEXT_MODELS_CACHE["model_info"]:
        try:
            response = requests.get("https://text.pollinations.ai/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if models_data and len(models_data) > 0:
                    # Store model info as {name, description} pairs
                    model_info = []
                    for model in models_data:
                        model_info.append({
                            'name': model['name'],
                            'description': model.get('description', f"Default {model['name']} model")
                        })
                    TEXT_MODELS_CACHE["model_info"] = model_info
                else:
                    TEXT_MODELS_CACHE["model_info"] = [
                        {'name': name, 'description': f"Default {name} model"}
                        for name in DEFAULT_TEXT_MODELS
                    ]
                TEXT_MODELS_CACHE["last_update"] = current_time
            else:
                TEXT_MODELS_CACHE["model_info"] = [
                    {'name': name, 'description': f"Default {name} model"}
                    for name in DEFAULT_TEXT_MODELS
                ]
        except Exception as e:
            print(f"Error fetching text models: {e}")
            TEXT_MODELS_CACHE["model_info"] = [
                {'name': name, 'description': f"Default {name} model"}
                for name in DEFAULT_TEXT_MODELS
            ]
    
    # Return only descriptions for display
    return [model['description'] for model in TEXT_MODELS_CACHE["model_info"]]

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
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🌸Pollinations"
    
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
            base_url = "https://image.pollinations.ai/prompt/"
            full_prompt = prompt
            if negative_prompt:
                full_prompt = f"{prompt} ### {negative_prompt}"
                
            encoded_prompt = quote(full_prompt)
            params = {
                "model": model,
                "width": width,
                "height": height,
            }
            
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
            
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}{encoded_prompt}?{param_str}"
            
            print(f"Generating image, URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            final_prompt = full_prompt
            
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
            
            temp_dir = tempfile.gettempdir()
            filename = f"pollinations_{int(time.time())}.png"
            image_path = os.path.join(temp_dir, filename)
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            image = Image.open(image_path)
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
            
            return (image_tensor, url, final_prompt)
            
        except Exception as e:
            error_msg = f"Pollinations API error: {str(e)}"
            print(error_msg)
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

class PollinationsTextGen:
    @classmethod
    def INPUT_TYPES(cls):
        text_models = get_text_models()
        default_description = next((model['description'] for model in TEXT_MODELS_CACHE.get("model_info", []) 
                                 if model['name'] == "openai"), text_models[0] if text_models else "Default openai model")
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter your text prompt..."}),
                "model": (text_models, {"default": default_description}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "private": ("BOOLEAN", {"default": True, "tooltip": "Keep the generation private"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "🧪AILab/🌸Pollinations"
    
    def generate_text(self, prompt, model, seed, private=True):
        try:
            # Find the model name that matches this description
            model_name = None
            for model_info in TEXT_MODELS_CACHE["model_info"]:
                if model_info['description'] == model:
                    model_name = model_info['name']
                    break
                
            if not model_name:
                model_name = "openai"  # fallback
            
            params = {
                "model": model_name,
                "seed": seed,
                "private": str(private).lower()
            }
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(prompt)}?{param_str}"
            
            print(f"Generating Text, URL: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                return (response.text,)
            else:
                return (f"Error: {response.status_code}",)
        except Exception as e:
            return (f"Text generation failed: {str(e)}",)

# Adding Text-to-Speech node
class PollinationsTextToSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "Enter text to convert to speech..."}),
                "voice": (["nova", "alloy", "echo", "fable", "onyx", "shimmer"], {"default": "nova"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "private": ("BOOLEAN", {"default": True, "tooltip": "Keep the generation private"})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "audio_path",)
    FUNCTION = "generate_speech"
    CATEGORY = "🧪AILab/🌸Pollinations"
    
    def generate_speech(self, text, voice, seed, private=True):
        try:
            params = {
                "model": "openai-audio",
                "voice": voice,
                "seed": seed,
                "private": str(private).lower()
            }
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(text)}?{param_str}"
            
            print(f"Generating Speech, URL: {url}")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Get ComfyUI's temp directory
                temp_dir = os.path.join(folder_paths.get_output_directory(), "pollinations_temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate unique filename
                timestamp = int(time.time())
                mp3_filename = f"pollinations_speech_{timestamp}.mp3"
                mp3_path = os.path.join(temp_dir, mp3_filename)
                
                # Save MP3 file
                with open(mp3_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Load and process audio
                waveform, sample_rate = torchaudio.load(mp3_path)
                
                # Ensure mono audio (take mean if stereo)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Add batch dimension if needed
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
                
                # Normalize audio
                if waveform.numel() > 0:
                    max_val = waveform.abs().max()
                    if max_val > 0:
                        waveform = waveform / max_val
                
                # Return audio in ComfyUI format
                audio_dict = {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }
                
                return (audio_dict, mp3_path)
            else:
                print(f"Error generating speech: {response.status_code}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")
        except Exception as e:
            error_msg = f"Speech generation failed: {str(e)}"
            print(error_msg)
            return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "PollinationsImageGen": PollinationsImageGen,
    "PollinationsTextGen": PollinationsTextGen,
    "PollinationsTextToSpeech": PollinationsTextToSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PollinationsImageGen": "Image Gen 🖼️ (Pollinations)",
    "PollinationsTextGen": "Text Gen 📝 (Pollinations)",
    "PollinationsTextToSpeech": "Text To Speech Chat 🔊 (Pollinations)",
} 