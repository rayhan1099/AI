# Hugging Face Inference Providers Setup

## 🚀 New Integration

Your chatbot now uses **Hugging Face Inference Providers** - a modern, OpenAI-compatible API that unifies 15+ inference partners under a single endpoint.

## ✨ Benefits

- ✅ **OpenAI-compatible** - Easy to use
- ✅ **Multiple providers** - Groq, Together AI, Novita, and more
- ✅ **Fast responses** - Optimized routing
- ✅ **No infrastructure** - Fully managed
- ✅ **Free tier available**

## 📝 Setup Instructions

### Step 1: Get Your Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Sign up or log in
3. Create a new token (read access is enough)
4. Copy your token (starts with `hf_`)

### Step 2: Configure Your Token

**Option A: Environment Variable (Recommended)**

```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"

# Windows (CMD)
set HF_TOKEN=your_token_here

# Linux/Mac
export HF_TOKEN="your_token_here"
```

**Option B: Update main.py**

Edit line 75 in `main.py`:
```python
HUGGINGFACE_API_KEY = "hf_your_actual_token_here"
```

**Option C: Create .env file**

Create a `.env` file in your project root:
```
HF_TOKEN=your_token_here
```

### Step 3: Choose a Model (Optional)

Default model: `moonshotai/Kimi-K2-Instruct-0905`

To change it, set environment variable:
```bash
export HF_MODEL="your-preferred-model"
```

Or update `main.py` line 77:
```python
HUGGINGFACE_MODEL = "your-preferred-model"
```

## 🎯 Available Models

You can use any model from Hugging Face Inference Providers:

- `moonshotai/Kimi-K2-Instruct-0905` (default - fast, good quality)
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `google/gemma-7b-it`
- And many more!

## 🧪 Test Your Setup

### Test 1: Check Configuration

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "Hello, how are you?",
    "conversation_id": "test"
  }'
```

### Test 2: Get Raw Response

```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "Write a short poem about AI",
    "conversation_id": "test"
  }'
```

## 📊 Response Format

### Standard Response
```json
{
  "user_message": "Hello",
  "ai_response": "Hello! How can I help you today?",
  "success": true,
  "source": "huggingface_inference_providers",
  "authenticated": true
}
```

### Raw Response (with return_raw=true)
```json
{
  "user_message": "Hello",
  "ai_response": "Hello! How can I help you today?",
  "raw_api_response": {
    "id": "chatcmpl-...",
    "model": "moonshotai/Kimi-K2-Instruct-0905",
    "choices": [...],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 15,
      "total_tokens": 25
    }
  },
  "metadata": {
    "model": "moonshotai/Kimi-K2-Instruct-0905",
    "provider": "groq",
    "tokens_used": 25
  },
  "success": true,
  "source": "huggingface_inference_providers"
}
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional
HF_MODEL=moonshotai/Kimi-K2-Instruct-0905  # Default model
USE_OLLAMA=False  # Set to True to use Ollama instead
```

### Code Configuration (main.py)

```python
# Line 75-77
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN", "YOUR_TOKEN_HERE")
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
HUGGINGFACE_MODEL = os.getenv("HF_MODEL", "moonshotai/Kimi-K2-Instruct-0905")
```

## 🎨 Provider Selection

Hugging Face automatically routes to the best available provider:
- **Groq** - Ultra-fast inference
- **Together AI** - High-quality models
- **Novita** - Cost-effective
- And 12+ more providers

You don't need to choose - it's automatic!

## ⚠️ Troubleshooting

### Error: "Hugging Face API key not configured"
- Make sure you set `HF_TOKEN` environment variable
- Or update `HUGGINGFACE_API_KEY` in `main.py`

### Error: "OpenAI package required"
- Install: `pip install openai`
- Or: `pip install -r requirements.txt`

### Error: Rate limit exceeded
- Free tier has rate limits
- Consider upgrading or using Ollama for unlimited local use

### Still using fallback responses?
- Check that `USE_OLLAMA = False`
- Verify your token is correct
- Check server logs for errors

## 💡 Tips

1. **Free Tier**: Hugging Face offers free tier with rate limits
2. **Model Selection**: Try different models to find what works best
3. **Caching**: Responses are cached for better performance
4. **Monitoring**: Check usage in Hugging Face dashboard

## 🔗 Resources

- **Hugging Face Tokens**: https://huggingface.co/settings/tokens
- **Inference Providers**: https://huggingface.co/inference-providers
- **Available Models**: https://huggingface.co/models
- **Documentation**: https://huggingface.co/docs/api-inference

## ✅ Quick Start Checklist

- [ ] Get Hugging Face token
- [ ] Set `HF_TOKEN` environment variable
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Restart server
- [ ] Test with a message
- [ ] Enjoy AI-powered responses! 🎉

