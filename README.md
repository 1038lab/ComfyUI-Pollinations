# ComfyUI-Pollinations

## Introduction

ComfyUI-Pollinations is a custom node for ComfyUI that utilizes the Pollinations API to generate images and text based on user prompts. This library provides two main functionalities: image generation and text generation, allowing users to create visual and textual content easily.

## Update
- **V1.3.0 (2025-07-06)**: 🔑 **Premium Models Support** - Added authentication-required models (gptimage, kontext) with separate premium nodes. Fixed text-to-speech to work without authentication. Added premium TTS node. See [PREMIUM_MODELS.md](PREMIUM_MODELS.md) for setup guide. ( [update.md](update.md#v130-20250706) )

![v1 3 0](https://github.com/user-attachments/assets/365fea76-7d35-4595-b4a1-de72416476b4)

- V1.2.1 (2025-06-19): requirments.txt update

- V1.2.0 (2025-05-31): Add Text to Speech feature using Pollinations API ( [update.md](update.md#v110-20250305) )

![v1 2 0 text to speech chat](<example_workflows/text to speech chat.png>)

- V1.1.0 (2025-03-21): Integrated internationalization (`i18n`) support for multiple languages. ( [update.md](update.md#v110-20250305) )

![v1 1 0](https://github.com/user-attachments/assets/c2391cc6-3284-4d14-aa91-7bb0145028db)
(This the `i18n` Demo with `Chinese` UI)

## Support Model List

#### Free Image Generation Models (No Authentication Required)

| Free Models | Description |
|-------------|-------------|
| flux        | High-quality image generation |
| turbo       | Ultra-fast image generation |

#### 🔑 Premium Image Models (Requires API Token)

| Premium Models | Description | Required Tier |
|----------------|-------------|---------------|
| gptimage       | Advanced image generation with enhanced prompt understanding | flower tier |
| kontext        | Professional image-to-image editing and enhancement | seed tier |

**Setup Premium Models**: Get API token from https://auth.pollinations.ai - See [PREMIUM_MODELS.md](PREMIUM_MODELS.md) for detailed guide.

| Text Generation Models       | Text Generation Models       | Text Generation Models       | Text Generation Models       |
|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| openai                        | openai-large                  | openai-reasoning              | qwen-coder                   |
| llama                         | mistral                       | unity                         | midijourney                   |
| rtist                         | searchgpt                     | evil                          | deepseek                      |
| claude-hybridspace            | deepseek-r1                   | deepseek-reasoner             | llamalight                    |
| llamaguard                    | gemini                        | gemini-thinking               | hormoz                        |
| hypnosis-tracy                | sur                           | sur-mistral                   | llama-scaleway                |

#### Text Generation Models

| Name                   | Type    | Censored | Description                                 | Base Model | Vision | Reasoning | Provider     |
|------------------------|--------|----------|---------------------------------------------|------------|--------|-----------|-------------|
| openai                | chat   | Yes      | OpenAI GPT-4o-mini                         | Yes        | Yes    | No        |             |
| openai-large          | chat   | Yes      | OpenAI GPT-4o                              | Yes        | Yes    | No        |             |
| openai-reasoning      | chat   | Yes      | OpenAI o1-mini                             | Yes        | No     | Yes       |             |
| qwen-coder           | chat   | Yes      | Qwen 2.5 Coder 32B                         | Yes        | No     | No        |             |
| llama                 | chat   | No       | Llama 3.3 70B                              | Yes        | No     | No        |             |
| mistral              | chat   | No       | Mistral Nemo                               | Yes        | No     | No        |             |
| unity                | chat   | No       | Unity with Mistral Large by Unity AI Lab   | No         | No     | No        |             |
| midijourney          | chat   | Yes      | Midijourney musical transformer            | No         | No     | No        |             |
| rtist                | chat   | Yes      | Rtist image generator by @bqrio            | No         | No     | No        |             |
| searchgpt           | chat   | Yes      | SearchGPT with realtime news and web search | No         | No     | No        |             |
| evil                 | chat   | No       | Evil Mode - Experimental                   | No         | No     | No        |             |
| deepseek            | chat   | Yes      | DeepSeek-V3                                | Yes        | No     | No        |             |
| claude-hybridspace  | chat   | Yes      | Claude Hybridspace                         | Yes        | No     | No        |             |
| deepseek-r1         | chat   | Yes      | DeepSeek-R1 Distill Qwen 32B               | Yes        | No     | Yes       | cloudflare  |
| deepseek-reasoner   | chat   | Yes      | DeepSeek R1 - Full                         | Yes        | No     | Yes       | deepseek    |
| llamalight          | chat   | No       | Llama 3.1 8B Instruct                      | Yes        | No     | No        |             |
| llamaguard          | safety | No       | Llamaguard 7B AWQ                          | No         | No     | No        | cloudflare  |
| gemini              | chat   | Yes      | Gemini 2.0 Flash                           | Yes        | No     | No        | google      |
| gemini-thinking     | chat   | Yes      | Gemini 2.0 Flash Thinking                  | Yes        | No     | No        | google      |
| hormoz              | chat   | No       | Hormoz 8b by Muhammadreza Haghiri          | No         | No     | No        | modal.com   |
| hypnosis-tracy      | chat   | No       | Hypnosis Tracy - Your Self-Help AI         | No         | No     | No        | modal.com   |
| sur                 | chat   | Yes      | Sur AI Assistant                           | No         | No     | No        |             |
| sur-mistral        | chat   | Yes      | Sur AI Assistant (Mistral)                 | No         | No     | No        |             |
| llama-scaleway      | chat   | No       | Llama (Scaleway)                           | Yes        | No     | No        |             |

### 1. PollinationsImageGen

- **Function**: Generates images based on a textual prompt.
- **Input Parameters**:
  - `prompt`: Description of the image to generate.
  - `model`: The model to use for image generation (e.g., "flux").
  - `width`: Width of the generated image.
  - `height`: Height of the generated image.
  - `batch_size`: Number of images to generate.
  - `negative_prompt`: Optional prompt to specify what to avoid in the image.
  - `seed`: Random seed for generation.
  - `enhance`: Whether to enhance the image.
  - `nologo`: Whether to include a logo.
  - `private`: Whether the generation is private.
  - `safe`: Whether to apply safety filters.

### 2. PollinationsTextGen

- **Function**: Generates text based on a textual prompt.
- **Input Parameters**:
  - `prompt`: The text prompt for generation.
  - `model`: The model to use for text generation (e.g., "openai").
  - `seed`: Random seed for generation.
  - `private`: Whether the generation is private.

### 3. PollinationsTextToSpeech

- **Function**: Converts text to speech audio using OpenAI's audio models through Pollinations API.
- **Input Parameters**:
  - `text`: The text to convert to speech.
  - `model`: Text-to-speech model ("openai-audio" or "hypnosis-tracy")
  - `voice`: The voice to use for speech generation (e.g., "nova", "alloy", "echo", "fable", "onyx", "shimmer", "coral", "verse", "ballad", "ash", "sage", "amuch", "dan").
  - `seed` (optional): Random seed for generation (requires authentication).
- **Output**:
  - `audio`: Audio file path that can be used with audio playback nodes.
- **Note**: Works without authentication by default. Seed parameter requires authentication.

## 💎 Premium Nodes (Authentication Required)

### 1. PollinationsPremiumImageGen 🔑

- **Function**: Enhanced image generation with premium models like GPTImage and Kontext.
- **Input Parameters**:
  - `api_token`: Your Pollinations API token (get from https://auth.pollinations.ai)
  - `prompt`: Text description of the image you want to generate.
  - `model`: Premium model to use ("gptimage" or "kontext")
  - `width`, `height`: Image dimensions (64-2048px)
  - `seed`: Random seed for reproducible results
  - `count`: Number of images to generate (1-4)
  - `enhance`: Enable prompt enhancement
  - `nologo`: Remove Pollinations logo
  - `private`: Keep generation private
  - `safe`: Enable safe mode
  - `transparent`: Generate with transparent background
- **Output**:
  - `images`: Generated images
  - `urls`: Direct URLs to the images
  - `prompts`: Enhanced prompts used for generation

### 2. PollinationsPremiumImageEdit 🔑

- **Function**: Advanced image editing and enhancement using premium models.
- **Input Parameters**:
  - `api_token`: Your Pollinations API token
  - `image`: Input image to edit
  - `prompt`: Description of desired changes
  - `model`: Premium model to use ("gptimage" or "kontext")
  - `width`, `height`: Output dimensions
  - `seed`: Random seed
  - `enhance`: Enable prompt enhancement
  - `nologo`: Remove Pollinations logo
  - `private`: Keep generation private
- **Output**:
  - `image`: Edited image
  - `url`: Direct URL to the edited image
  - `prompt`: Enhanced prompt used for editing

### 3. PollinationsPremiumTextToSpeech 🔑

- **Function**: Premium text-to-speech with full feature access including seed control and private mode.
- **Input Parameters**:
  - `api_token`: Your Pollinations API token (requires "seed tier" or higher)
  - `text`: The text to convert to speech.
  - `model`: Text-to-speech model ("openai-audio" or "hypnosis-tracy")
  - `voice`: The voice to use for speech generation (13 voices available)
  - `seed` (optional): Random seed for reproducible results
  - `private` (optional): Keep the generation private (default: true)
- **Output**:
  - `audio`: High-quality audio file
  - `filename`: Generated filename
- **Note**: Provides full access to all text-to-speech features with authentication.

### Image Feed and Text Feed Model

#### Image Feed Model

#### Image Feed Model
The Image Feed Model is designed to retrieve and generate images based on user-defined parameters. It allows users to specify the number of images to fetch, the model to use for generation, and optional filters such as prompts and refresh settings. This model is particularly useful for applications that require dynamic image generation based on varying inputs.

#### Text Feed Model
The Text Feed Model is focused on generating text responses based on user prompts. It supports various text generation models and allows users to specify the number of responses to retrieve, along with optional filters for model selection and prompt matching. This model is ideal for applications that need to generate conversational or informative text based on user queries.

## Installation

To install Pollinations, you can clone the repository and add it to your ComfyUI custom nodes directory. Ensure you have the required dependencies installed.

### Method 1. install on ComfyUI-Manager, search `ComfyUI-Pollinations` and install
install requirment.txt in the ComfyUI-Pollinations
  ```bash
  ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
  ```

### Method 2. Clone this repository to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-Pollinations.git
```
install requirment.txt in the ComfyUI-Pollinations folder
```bash
/ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

## Usage

After installation, you can use the nodes in your ComfyUI workflow. Simply drag and drop the `PollinationsImageGen` or `PollinationsTextGen` nodes into your canvas and configure the input parameters as needed.

### PollinationsImageGen Node
![ImageGen](https://github.com/user-attachments/assets/508a08c0-df49-4a18-9e8a-5c1be10084db)
![ImageGen_2](https://github.com/user-attachments/assets/82354742-c91b-466c-b913-dbf78e587b9e)
Generate 4 images simultaneously

### PollinationsTextGen Node
![TextGen](https://github.com/user-attachments/assets/30f774c4-c0b4-4122-aede-4c6f47be6721)

![TextGen_2](https://github.com/user-attachments/assets/a2069c7a-e4c0-4581-a2cb-96d532adb04b)

## Contributing

We welcome contributions to Pollinations! Please fork the repository and submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to all contributors and users who have supported the development of ComfyUI-Pollinations.

If this custom node helps you or you like Our work, please give me ⭐ on this repo!

It's a great encouragement for my efforts!
