## Version History

### V1.3.0 (2025/07/06)

![v1 3 0](https://github.com/user-attachments/assets/365fea76-7d35-4595-b4a1-de72416476b4)

- **🔑 Premium Models Support**: Added authentication-required models (gptimage, kontext)
- **📁 Separated Premium Nodes**: Created `pollinations_premium.py` for tier-based models
- **🎨 Advanced Image Editing**: Premium image-to-image editing with kontext model
- **🔧 Code Optimization**: Streamlined main nodes to only include free models (flux, turbo)
- **📖 Enhanced Documentation**: Added comprehensive premium models guide
- **⚡ Improved Performance**: Removed authentication checks from free model workflows
- **🛡️ Better Error Handling**: Clear authentication error messages and tier upgrade guidance
- **🔍 Search Node**: Added web search functionality with `elixposearch` model
- **🚫 Removed Basic TTS**: Removed PollinationsTextToSpeech node due to severe API limitations
- **🔊 Premium TTS Only**: Text-to-speech now only available through premium node with API authentication

**New Premium Nodes** (requires API token from https://auth.pollinations.ai):
- `Premium Image Gen 🔑`: Advanced image generation with gptimage/kontext
- `Premium Image Edit 🔑`: Professional image-to-image editing capabilities
- `Premium Text-to-Speech 🔑`: Full TTS features with unrestricted text, seed control and private mode

**New Node**:
- `Search 🔍`: Web search functionality

**⚠️ BREAKING CHANGE - Text-to-Speech**:
- **Removed**: Basic PollinationsTextToSpeech node (was too limited to be useful)
- **Reason**: Free tier only supported ~10 predefined words ("hi", "hello", "yes", "good", "cat", "dog", "test")
- **Solution**: Use `Premium Text-to-Speech 🔑` node with API token for full functionality
- **Migration**: Get API token from https://auth.pollinations.ai and use premium node instead

**Free Models** (no authentication required):
- `flux`: High-quality image generation
- `turbo`: Ultra-fast image generation

📋 **Setup Guide**: See [PREMIUM_MODELS.md](PREMIUM_MODELS.md) for detailed authentication and usage instructions.

### V1.2.0 (2025/05/30)
- Added Text to Speech feature using Pollinations API
- Support for multiple voice options: nova, alloy, echo, fable, onyx, shimmer
- Audio output as file path for integration with other audio nodes

![v1 2 0 text to speech chat](<example_workflows/text to speech chat.png>)

### V1.1.0 (2025/03/05)
- Added Image Feed and Text Feed Custom Nodes
- Real-time feed of Pollinations AI generated content
- Filter capabilities by model and prompt content
- Refresh option for getting latest content

![v1 1 0](https://github.com/user-attachments/assets/c2391cc6-3284-4d14-aa91-7bb0145028db)

### V1.0.0 (2025/02/15)
- Initial release
- Support for image generation with various models
- Text generation capabilities
- ComfyUI Manager compatibility
- Custom node integration with ComfyUI workflow

## Upcoming Features

- [ ] Speech-to-Text functionality
- [ ] Enhanced audio options (pitch, speed)
- [ ] Batch audio generation
- [ ] Custom voice fine-tuning integration

## Fixes

- [x] Fixed ComfyUI Manager installation issues
- [x] Improved error handling for API validation
- [x] Updated model list to include latest Pollinations models
- [x] Optimized temporary file management 
