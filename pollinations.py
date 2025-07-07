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

DEFAULT_IMAGE_MODELS = ["flux", "turbo"]

TEXT_GENERATION_MODELS = [
    "openai",           # OpenAI GPT-4o Mini
    "mistral",          # Mistral Small 3.1 24B
    "llamascout",       # Llama 4 Scout 17B
    "openai-fast",      # OpenAI GPT-4.1 Nano
    "openai-reasoning", # OpenAI O3
    "phi",              # Phi-4 Mini Instruct
    "qwen-coder",       # Qwen 2.5 Coder 32B
    "bidara",           # NASA BIDARA
    "midijourney"       # MIDIjourney
]

SEARCH_MODELS = [
    "searchgpt",        # OpenAI GPT-4o Mini Search Preview
    "elixposearch"      # Elixpo Search
]

TEXT_TO_SPEECH_MODELS = [
    "openai-audio",     # OpenAI GPT-4o Mini Audio Preview
    "hypnosis-tracy"    # Hypnosis Tracy
]

# Updated voice list from API
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage", "amuch", "dan"]
class PollinationsImageGen:

    @classmethod
    def INPUT_TYPES(cls):
        # Use fixed model list

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter a description of the image you want..."}),
                "model": (DEFAULT_IMAGE_MODELS, {"default": "flux"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
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
    
    def generate(self, prompt, model, width, height, batch_size=1, seed=0,
                 enhance=True, nologo=True, private=True, safe=False):
        """Generate multiple images"""
        # Use fixed model list

        images = []
        urls = []
        prompts = []

        for i in range(batch_size):
            current_seed = seed + i if seed != 0 else 0
            try:
                image, url, final_prompt = self._generate_single(
                    prompt, model, width, height,
                    current_seed, enhance, nologo, private, safe
                )
                images.append(image)
                urls.append(url)
                prompts.append(final_prompt)
            except Exception as e:
                images.append(torch.zeros(1, 512, 512, 3))
                urls.append(f"Error: {str(e)}")
                prompts.append(prompt)

        return (images, urls, prompts)
    
    def _generate_single(self, prompt, model, width, height, seed=0,
                        enhance=True, nologo=True, private=True, safe=False):
        """Generate a single image"""
        try:
            base_url = "https://image.pollinations.ai/prompt/"
            encoded_prompt = quote(prompt)
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

            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            final_prompt = prompt

            try:
                image_url = response.url
                if "/prompt/" in image_url:
                    encoded_part = image_url.split("/prompt/")[1].split("?")[0]
                    extracted_prompt = unquote(encoded_part)
                    if extracted_prompt != prompt and enhance:
                        final_prompt = extracted_prompt
            except Exception:
                pass
            
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
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()


class PollinationsTextGen:
    @classmethod
    def INPUT_TYPES(cls):
        # Use fixed model list
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter your text prompt..."}),
                "model": (TEXT_GENERATION_MODELS, {"default": "openai"}),
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
            # Use model directly from fixed list
            model_name = model if model in TEXT_GENERATION_MODELS else "openai"
            
            params = {
                "model": model_name,
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

# Adding Search node
class PollinationsSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"multiline": True, "placeholder": "Enter your search query..."}),
                "model": (SEARCH_MODELS, {"default": "searchgpt"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "private": ("BOOLEAN", {"default": True, "tooltip": "Keep the search private"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("search_results",)
    FUNCTION = "search"
    CATEGORY = "🧪AILab/🌸Pollinations"

    def search(self, query, model, seed=0, private=True):
        try:
            params = {
                "model": model,
                "seed": seed,
                "private": str(private).lower()
            }
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(query)}?{param_str}"

            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return (response.text,)
            else:
                return (f"Search error: {response.status_code}",)
        except Exception as e:
            return (f"Search failed: {str(e)}",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# Adding Text-to-Speech node
class PollinationsTextToSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": "Enter text to convert to speech..."}),
                "model": (TEXT_TO_SPEECH_MODELS, {"default": "openai-audio"}),
                "voice": (AVAILABLE_VOICES, {"default": "nova"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed (requires authentication)"})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "audio_path",)
    FUNCTION = "generate_speech"
    CATEGORY = "🧪AILab/🌸Pollinations"
    
    def generate_speech(self, text, model, voice, seed=None):
        try:
            # Use selected model, fallback to openai-audio if not in list
            model_name = model if model in TEXT_TO_SPEECH_MODELS else "openai-audio"

            params = {
                "model": model_name,
                "voice": voice
            }

            # Only add seed if provided and not 0 (requires authentication)
            if seed is not None and seed != 0:
                params["seed"] = seed

            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(text)}?{param_str}"

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
            elif response.status_code == 402:
                # Payment required - authentication needed (usually when using seed)
                error_msg = "Text-to-Speech with seed parameter requires authentication. Remove seed or visit https://auth.pollinations.ai to get authentication."
                print(f"[PollinationsTextToSpeech] {error_msg}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")
            else:
                print(f"[PollinationsTextToSpeech] Error: HTTP {response.status_code} - {response.text[:200]}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")
        except Exception as e:
            print(f"[PollinationsTextToSpeech] Exception: {str(e)}")
            return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "PollinationsImageGen": PollinationsImageGen,
    "PollinationsTextGen": PollinationsTextGen,
    "PollinationsSearch": PollinationsSearch,
    "PollinationsTextToSpeech": PollinationsTextToSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PollinationsImageGen": "Image Gen 🖼️ (Pollinations)",
    "PollinationsTextGen": "Text Gen 📝 (Pollinations)",
    "PollinationsSearch": "Search 🔍 (Pollinations)",
    "PollinationsTextToSpeech": "Text To Speech Chat 🔊 (Pollinations)",
}