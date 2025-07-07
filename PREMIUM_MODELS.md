# Pollinations Premium Models

## ⚠️ Text-to-Speech Migration (v1.3.0)
**BREAKING CHANGE**: The basic `PollinationsTextToSpeech` node has been removed due to severe API limitations (only ~10 predefined words supported). Use `Premium Text-to-Speech 🔑` node instead for unlimited text conversion.

## Overview
Premium models require authentication and tier-based access from Pollinations.ai. These models offer enhanced capabilities but need API tokens.

## Available Premium Models

### 🔑 Premium Image Generation
- **gptimage**: Advanced image generation (requires "flower tier")
- **kontext**: Image-to-image editing and enhancement (requires "seed tier")

### 🔑 Premium Text-to-Speech (Authentication Required)
- **openai-audio**: OpenAI GPT-4o Mini Audio Preview (requires API token)
- **hypnosis-tracy**: Hypnosis Tracy voice model (requires API token)

## How to Get Access

1. **Visit Authentication Page**: https://auth.pollinations.ai
2. **Create Account**: Sign up or log in
3. **Request Tier Access**:
   - For gptimage: Request "flower tier" access
   - For kontext: Request "seed tier" access
   - For text-to-speech: Request "seed tier" access
4. **Get API Token**: Copy your API token from the dashboard

## Using Premium Nodes

### Premium Image Generator 🔑
- **Node**: `PollinationsPremiumImageGen`
- **Required**: API token, prompt, model selection
- **Models**: gptimage, kontext
- **Features**: Enhanced prompting, higher quality output

### Premium Image Editor 🔑
- **Node**: `PollinationsPremiumImageEdit`
- **Required**: Input image, API token, prompt, model selection
- **Models**: gptimage, kontext
- **Features**: Advanced image-to-image editing

### Premium Text-to-Speech 🔑
- **Node**: `PollinationsPremiumTextToSpeech`
- **Required**: API token, text input
- **Models**: openai-audio, hypnosis-tracy
- **Features**: Unlimited text conversion, seed control, private mode, multiple voice options
- **Note**: Replaces the removed basic text-to-speech node

## Setup Instructions

1. **Install Nodes**: The premium nodes are automatically loaded with the main plugin
2. **Get API Token**: Follow the authentication process above
3. **Add Token**: Enter your API token in the node's "api_token" field
4. **Select Model**: Choose the appropriate premium model
5. **Generate**: Create your content with enhanced capabilities

## Error Messages

- **"API token required"**: You need to enter a valid API token
- **"Authentication failed"**: Check your API token is correct
- **"Requires higher tier access"**: Your account needs tier upgrade
- **"tier access"**: Model requires specific tier level

## Free vs Premium

### Free Models (No Token Required)
- **flux**: Fast, good quality image generation
- **turbo**: Ultra-fast image generation

### Premium Models (Authentication Required)
- **gptimage**: Enhanced image generation with better prompt understanding
- **kontext**: Advanced image editing and enhancement capabilities
- **openai-audio**: High-quality text-to-speech with multiple voices
- **hypnosis-tracy**: Alternative text-to-speech model

## Support

For authentication issues or tier access requests:
- Visit: https://auth.pollinations.ai
- Contact Pollinations support for tier upgrades
- Check your account dashboard for token management

## Notes

- Premium models have higher quality but require authentication
- Free models work without any setup
- API tokens are account-specific and should be kept secure
- Different models require different tier levels
