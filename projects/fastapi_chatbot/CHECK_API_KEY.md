# 🔍 Check Your API Key Status

## Current Status: ❌ **NOT SET**

Your Hugging Face API key is **not configured** yet.

## 📋 How to Check

### Method 1: Check Environment Variables
```bash
# Windows PowerShell
echo $env:HF_TOKEN

# Windows CMD
echo %HF_TOKEN%

# Linux/Mac
echo $HF_TOKEN
```

If it shows nothing or is empty → **NOT SET**

### Method 2: Check main.py
Look at line 83 in `main.py`:
```python
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY", "YOUR_FREE_API_KEY"))
```

If it's using the default `"YOUR_FREE_API_KEY"` → **NOT SET**

### Method 3: Test the API
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

If response shows `"source": "fallback"` → **NOT SET**

## ✅ How to Set It

### Option 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:HF_TOKEN="hf_your_actual_token_here"
```

**Windows CMD:**
```cmd
set HF_TOKEN=hf_your_actual_token_here
```

**Linux/Mac:**
```bash
export HF_TOKEN="hf_your_actual_token_here"
```

**To make it permanent (Windows):**
1. Open System Properties → Environment Variables
2. Add new variable: `HF_TOKEN` = `your_token_here`

**To make it permanent (Linux/Mac):**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export HF_TOKEN="hf_your_actual_token_here"
```

### Option 2: Direct in main.py

Edit line 83 in `main.py`:
```python
HUGGINGFACE_API_KEY = "hf_your_actual_token_here"  # Replace with your actual token
```

### Option 3: .env File

Create a `.env` file in your project root:
```
HF_TOKEN=hf_your_actual_token_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

And load it in main.py:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 🔑 Get Your Token

1. Go to: https://huggingface.co/settings/tokens
2. Sign up or log in
3. Click "New token"
4. Name it (e.g., "chatbot")
5. Select "Read" access
6. Copy the token (starts with `hf_`)

## 🧪 Verify It's Set

After setting your token, test it:

```bash
# Test 1: Check environment variable
python -c "import os; print('Token:', os.getenv('HF_TOKEN', 'NOT SET')[:20] + '...' if os.getenv('HF_TOKEN') else 'NOT SET')"

# Test 2: Test API endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

If you see `"source": "huggingface_inference_providers"` → ✅ **SET CORRECTLY**

## 📊 Current Configuration

```
API Key Status: ❌ NOT SET
Current Value: YOUR_FREE_API_KEY (placeholder)
Using: Fallback responses
```

## 🎯 Next Steps

1. **Get token** from https://huggingface.co/settings/tokens
2. **Set environment variable** (see above)
3. **Restart server** to load the new token
4. **Test** - should see real AI responses!

## ⚠️ Important Notes

- Token should start with `hf_`
- Keep your token secret - don't commit it to git
- Use environment variables for security
- Restart server after setting token

