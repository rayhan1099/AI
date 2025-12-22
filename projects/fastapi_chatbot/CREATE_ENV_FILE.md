# 📝 Create .env File - Step by Step

## ✅ Setup Complete!

I've prepared everything for you. Now you just need to create the `.env` file manually.

## 🎯 Quick Steps

### Step 1: Create the .env File

In your project root directory (`fastapi_chatbot`), create a new file named `.env` (with the dot at the beginning).

**Windows:**
- Right-click in the folder → New → Text Document
- Rename it to `.env` (make sure to remove `.txt` extension)
- If Windows asks about changing the extension, click "Yes"

**Or use command line:**
```bash
# In your project directory
echo HF_TOKEN=YOUR_HUGGINGFACE_API_KEY > .env
```

### Step 2: Add Your Token

Open the `.env` file and add this content:

```env
# Hugging Face API Configuration
HF_TOKEN=YOUR_HUGGINGFACE_API_KEY

# Optional: Specify a different model
# HF_MODEL=moonshotai/Kimi-K2-Instruct-0905

# Optional: Use Ollama instead (set to true if Ollama is running)
# USE_OLLAMA=False
```

### Step 3: Save and Restart

1. Save the `.env` file
2. Restart your server:
   ```bash
   uvicorn main:app --reload
   ```

## 📋 Complete .env File Content

Copy and paste this into your `.env` file:

```env
# Hugging Face API Configuration
HF_TOKEN=YOUR_HUGGINGFACE_API_KEY

# Optional: Specify a different model
# HF_MODEL=moonshotai/Kimi-K2-Instruct-0905

# Optional: Use Ollama instead (set to true if Ollama is running)
# USE_OLLAMA=False

# API Keys for authentication (optional - for custom keys)
# API_KEY_1=your-custom-key-1
# API_KEY_2=your-custom-key-2
```

## ✅ What I've Done

1. ✅ Added `python-dotenv` to `requirements.txt`
2. ✅ Installed `python-dotenv` package
3. ✅ Updated `main.py` to load `.env` file automatically
4. ✅ Created `.gitignore` to protect your `.env` file
5. ✅ Created `.env.example` as a template

## 🔒 Security

- ✅ `.env` is now in `.gitignore` - won't be committed to git
- ✅ Your token is safe and won't be exposed
- ✅ You can share `.env.example` without your actual token

## 🧪 Verify It Works

After creating `.env` and restarting:

```bash
# Test the API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

You should see:
- ✅ `"source": "huggingface_inference_providers"` (not "fallback")
- ✅ Real AI responses

## 📍 File Location

Your `.env` file should be here:
```
fastapi_chatbot/
├── .env          ← Create this file here
├── .env.example  ← Template (already created)
├── .gitignore    ← Already configured
├── main.py
└── ...
```

## 🎉 You're Done!

Once you create the `.env` file with your token, restart the server and everything will work automatically!

