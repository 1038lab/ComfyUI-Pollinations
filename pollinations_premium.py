import torch
import requests
import tempfile
import os
import time
from PIL import Image
import numpy as np
from urllib.parse import quote, unquote
import base64
import torchaudio
import folder_paths

# Premium models that require authentication
PREMIUM_IMAGE_MODELS = ["gptimage", "kontext"]
TEXT_TO_SPEECH_MODELS = ["openai-audio", "hypnosis-tracy"]
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage", "amuch", "dan"]

class PollinationsPremiumImageGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful landscape"}),
                "model": (PREMIUM_IMAGE_MODELS, {"default": "gptimage"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "enhance": ("BOOLEAN", {"default": True}),
                "nologo": ("BOOLEAN", {"default": True}),
                "private": ("BOOLEAN", {"default": True}),
                "safe": ("BOOLEAN", {"default": False}),
                "transparent": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls", "prompts")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🌸Pollinations/💎Premium"

    def generate(self, prompt, model, api_token, width=512, height=512, seed=0, count=1,
                 enhance=True, nologo=True, private=True, safe=False, transparent=False):
        
        if not api_token.strip():
            error_msg = f"API token required for {model} model. Get your token at: https://auth.pollinations.ai"
            empty_image = torch.zeros(1, 512, 512, 3)
            return ([empty_image], [error_msg], [prompt])

        images = []
        urls = []
        prompts = []

        for i in range(count):
            try:
                image, url, final_prompt = self._generate_single(
                    prompt, model, api_token, width, height, seed + i,
                    enhance, nologo, private, safe, transparent
                )
                images.append(image)
                urls.append(url)
                prompts.append(final_prompt)
            except Exception as e:
                images.append(torch.zeros(1, 512, 512, 3))
                urls.append(f"Error: {str(e)}")
                prompts.append(prompt)

        return (images, urls, prompts)

    def _generate_single(self, prompt, model, api_token, width, height, seed,
                        enhance=True, nologo=True, private=True, safe=False, transparent=False):
        try:
            base_url = "https://image.pollinations.ai/prompt/"

            encoded_prompt = quote(prompt)
            
            params = {
                "model": model,
                "width": str(width),
                "height": str(height),
                "seed": str(seed),
                "nologo": str(nologo).lower(),
                "private": str(private).lower()
            }
            
            if enhance and model in ["gptimage"]:
                params["enhance"] = "true"
            
            if safe:
                params["safe"] = "true"

            if transparent and model == "gptimage":
                params["transparent"] = "true"

            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}{encoded_prompt}?{param_str}"

            headers = {"Authorization": f"Bearer {api_token}"}
            response = requests.get(url, stream=True, headers=headers, timeout=60)
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
            filename = f"pollinations_premium_{int(time.time())}.png"
            image_path = os.path.join(temp_dir, filename)
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            image = Image.open(image_path)
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            
            os.unlink(image_path)
            
            return (image_tensor, url, final_prompt)
            
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                error_msg = f"Authentication failed. Check your API token for {model} model."
            elif "tier" in str(e).lower():
                error_msg = f"Model {model} requires higher tier access. Upgrade at: https://auth.pollinations.ai"
            else:
                error_msg = f"Premium API error: {str(e)}"
            
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class PollinationsPremiumImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_token": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "enhance this image"}),
                "model": (PREMIUM_IMAGE_MODELS, {"default": "kontext"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "enhance": ("BOOLEAN", {"default": True}),
                "nologo": ("BOOLEAN", {"default": True}),
                "private": ("BOOLEAN", {"default": True}),
                "safe": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "url", "prompt")
    FUNCTION = "edit_image"
    CATEGORY = "🧪AILab/🌸Pollinations/💎Premium"

    def edit_image(self, image, prompt, model, api_token, width=512, height=512, seed=0,
                   enhance=True, nologo=True, private=True, safe=False):
        
        if not api_token.strip():
            error_msg = f"API token required for {model} model. Get your token at: https://auth.pollinations.ai"
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

        try:
            image_np = (image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            max_size = 512
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            temp_dir = tempfile.gettempdir()
            input_filename = f"pollinations_premium_input_{int(time.time())}.jpg"
            input_path = os.path.join(temp_dir, input_filename)
            pil_image.save(input_path, "JPEG", quality=85)

            with open(input_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            os.unlink(input_path)

            image_url = f"data:image/jpeg;base64,{img_base64}"

            if len(image_url) > 8000:
                pass

            edited_image, result_url, final_prompt = self._edit_single(
                prompt, model, api_token, width, height, image_url,
                seed, enhance, nologo, private, safe
            )

            return (edited_image, result_url, final_prompt)

        except Exception as e:
            error_msg = f"Premium image editing error: {str(e)}"
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    def _edit_single(self, prompt, model, api_token, width, height, image_url, seed=0,
                     enhance=True, nologo=True, private=True, safe=False):
        try:
            base_url = "https://image.pollinations.ai/prompt/"

            encoded_prompt = quote(prompt)
            
            params = {
                "model": model,
                "width": str(width),
                "height": str(height),
                "seed": str(seed),
                "image": image_url,
                "nologo": str(nologo).lower(),
                "private": str(private).lower()
            }
            
            if enhance and model in ["kontext"]:
                params["enhance"] = "true"
            
            if safe:
                params["safe"] = "true"

            param_str = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
            url = f"{base_url}{encoded_prompt}?{param_str}"

            headers = {"Authorization": f"Bearer {api_token}"}
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()

            final_prompt = prompt

            temp_dir = tempfile.gettempdir()
            filename = f"pollinations_premium_edit_{int(time.time())}.png"
            image_path = os.path.join(temp_dir, filename)
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            edited_image = Image.open(image_path)
            edited_array = np.array(edited_image).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_array).unsqueeze(0)
            
            os.unlink(image_path)

            return (edited_tensor, url, final_prompt)

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                error_msg = f"Authentication failed. Check your API token for {model} model."
            elif "tier" in str(e).lower():
                error_msg = f"Model {model} requires higher tier access. Upgrade at: https://auth.pollinations.ai"
            else:
                error_msg = f"Premium editing API error: {str(e)}"
            
            empty_image = torch.zeros(1, 512, 512, 3)
            return (empty_image, error_msg, prompt)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class PollinationsPremiumTextToSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "text": ("STRING", {"multiline": True, "placeholder": "Enter text to convert to speech..."}),
                "model": (TEXT_TO_SPEECH_MODELS, {"default": "openai-audio"}),
                "voice": (AVAILABLE_VOICES, {"default": "nova"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "private": ("BOOLEAN", {"default": True, "tooltip": "Keep the generation private"})
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "filename")
    FUNCTION = "generate_speech"
    CATEGORY = "🧪AILab/🌸Pollinations/💎Premium"

    def generate_speech(self, api_token, text, model, voice, seed=0, private=True):
        if not api_token.strip():
            error_msg = f"API token required for premium text-to-speech. Get your token at: https://auth.pollinations.ai"
            print(f"[PollinationsPremiumTextToSpeech] {error_msg}")
            return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

        try:
            # Use selected model, fallback to openai-audio if not in list
            model_name = model if model in TEXT_TO_SPEECH_MODELS else "openai-audio"

            params = {
                "model": model_name,
                "voice": voice,
                "seed": seed
            }

            # Add private parameter for premium features
            if private:
                params["private"] = "true"

            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"https://text.pollinations.ai/{quote(text)}?{param_str}"

            headers = {
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "ComfyUI-Pollinations/1.3.0"
            }

            print(f"[PollinationsPremiumTextToSpeech] Generating speech with model: {model_name}, voice: {voice}")

            response = requests.get(url, headers=headers, stream=True)

            if response.status_code == 200:
                # Save the audio file
                output_dir = folder_paths.get_temp_directory()
                timestamp = int(time.time())
                filename = f"pollinations_tts_premium_{timestamp}.mp3"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Load audio using torchaudio
                waveform, sample_rate = torchaudio.load(filepath)

                # Convert to the format expected by ComfyUI
                audio_dict = {
                    "waveform": waveform.unsqueeze(0),  # Add batch dimension
                    "sample_rate": sample_rate
                }

                print(f"[PollinationsPremiumTextToSpeech] Audio generated successfully: {filename}")
                return (audio_dict, filename)

            elif response.status_code == 401:
                error_msg = "Invalid API token. Please check your token at https://auth.pollinations.ai"
                print(f"[PollinationsPremiumTextToSpeech] {error_msg}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

            elif response.status_code == 402:
                error_msg = "Insufficient tier access. Premium text-to-speech requires 'seed' tier or higher."
                print(f"[PollinationsPremiumTextToSpeech] {error_msg}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                print(f"[PollinationsPremiumTextToSpeech] {error_msg}")
                return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

        except Exception as e:
            error_msg = f"Error generating speech: {str(e)}"
            print(f"[PollinationsPremiumTextToSpeech] {error_msg}")
            return ({"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}, "")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "PollinationsPremiumImageGen": PollinationsPremiumImageGen,
    "PollinationsPremiumImageEdit": PollinationsPremiumImageEdit,
    "PollinationsPremiumTextToSpeech": PollinationsPremiumTextToSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PollinationsPremiumImageGen": "Premium Image Gen 🔑 (Pollinations)",
    "PollinationsPremiumImageEdit": "Premium Image Edit 🔑 (Pollinations)",
    "PollinationsPremiumTextToSpeech": "Premium Text-to-Speech 🔑 (Pollinations)",
}
