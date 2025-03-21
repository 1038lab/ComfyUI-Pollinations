# ComfyUI-Pollinations

## Introduction

ComfyUI-Pollinations is a custom node for ComfyUI that utilizes the Pollinations API to generate images and text based on user prompts. This library provides two main functionalities: image generation and text generation, allowing users to create visual and textual content easily.

$$\textcolor{red}{\Huge \text{If this custom node helps you or you like Our work, please give me ⭐ on this repo!}}$$  
$$\textcolor{red}{\Huge \text{It's a great encouragement for my efforts!}}$$

## Update
- V1.1.0 (2025-03-21): Integrated internationalization (`i18n`) support for multiple languages.

![v1 1 0](https://github.com/user-attachments/assets/c2391cc6-3284-4d14-aa91-7bb0145028db)

## Nodes Overview

### ![ComfyUI-pollinations](https://github.com/user-attachments/assets/017f69c4-c3e6-4243-8de9-23053b4b0ffd)


Support Model List

  | Image Generation Models       |
  |-------------------------------|
  | flux                          |
  | turbo                         |

  | Text Generation Models       | Text Generation Models       | Text Generation Models       | Text Generation Models       |
  |-------------------------------|-------------------------------|-------------------------------|-------------------------------|
  | openai                        | openai-large                  | openai-reasoning              | qwen-coder                   |
  | llama                         | mistral                       | unity                         | midijourney                   |
  | rtist                         | searchgpt                     | evil                          | deepseek                      |
  | claude-hybridspace            | deepseek-r1                   | deepseek-reasoner             | llamalight                    |
  | llamaguard                    | gemini                        | gemini-thinking               | hormoz                        |
  | hypnosis-tracy                | sur                           | sur-mistral                   | llama-scaleway                |

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
