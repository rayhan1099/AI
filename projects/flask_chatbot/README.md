# Flask Chatbot - Modern AI Chat Interface

A beautiful, modern Flask-based chatbot application with Hugging Face AI integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

The application uses a `.env` file for configuration. A `.env` file has already been created with your Hugging Face token configured.

**Important:** Make sure your `.env` file exists and contains:
```env
HF_TOKEN=your_huggingface_token_here
```

If you need to update the token or other settings, edit the `.env` file.

### Step 3: Run the Application

#### Option 1: Direct Python (Development)
```bash
python app.py
```

#### Option 2: Using Flask CLI
```bash
flask run
```

#### Option 3: With Custom Host/Port
```bash
python app.py
# Or set environment variables:
# Windows: set FLASK_RUN_HOST=0.0.0.0 && set FLASK_RUN_PORT=5000 && flask run
# Linux/Mac: FLASK_RUN_HOST=0.0.0.0 FLASK_RUN_PORT=5000 flask run
```

### Step 4: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

The chatbot interface will load and you can start chatting!

## 📁 Project Structure

```
flask_chatbot/
├── app.py              # Flask application (main server)
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (your config)
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
├── static/            # Static files (CSS, JS)
│   ├── css/
│   │   └── style.css  # Modern dark theme styles
│   └── js/
│       └── chat.js    # Chat functionality
└── templates/         # HTML templates
    └── index.html     # Main chat interface
```

## 🎯 Features

- ✨ **Modern Dark UI** - Beautiful gradient-based dark theme
- 🤖 **Hugging Face AI** - Powered by DialoGPT-medium model
- 💬 **Real-time Chat** - Smooth messaging experience
- 📝 **Conversation History** - Maintains chat context
- 📱 **Responsive Design** - Works on desktop and mobile
- ⚡ **Fast & Lightweight** - Optimized performance
- 🔒 **Secure** - Environment-based configuration
- 🎨 **Smooth Animations** - Polished user experience

## ⚙️ Configuration

All configuration is done through the `.env` file:

```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium

# AI Service Configuration
USE_OLLAMA=False
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama2

# Rate Limiting
RATE_LIMIT_PER_DAY=200
RATE_LIMIT_PER_HOUR=50
RATE_LIMIT_PER_MINUTE=10

# Content Limits
MAX_CONTENT_LENGTH=16777216
MAX_MESSAGE_LENGTH=1000

# Conversation Settings
MAX_CONVERSATION_HISTORY=10
```

## 🛠️ Troubleshooting

### "AI service is temporarily unavailable" Error

**This is the most common issue!** Hugging Face has deprecated their old Inference API endpoints. Here are solutions:

#### Solution 1: Use Hugging Face Inference Endpoints (Recommended)
1. Go to [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints)
2. Create a new endpoint with a conversational model (e.g., `microsoft/DialoGPT-medium`)
3. Update your `.env` file with the new endpoint URL:
   ```env
   HUGGINGFACE_API_URL=https://your-endpoint-id.region.inference.endpoints.huggingface.cloud
   ```

#### Solution 2: Use Alternative Free AI Services
Update `.env` to use a different service:
- **OpenAI API** (requires API key): Update code to use OpenAI
- **Anthropic Claude** (requires API key): Update code to use Claude
- **Local Ollama**: Set `USE_OLLAMA=True` in `.env` and install [Ollama](https://ollama.ai)

#### Solution 3: Use Hugging Face Spaces API
Some models are available via Spaces. Check [Hugging Face Spaces](https://huggingface.co/spaces) for available APIs.

### Port Already in Use
If port 5000 is already in use, change it in `.env`:
```env
PORT=5001
```

### Module Not Found Error
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Hugging Face API Errors
- Verify your `HF_TOKEN` in `.env` is correct and valid
- Check your internet connection
- The API might be rate-limited (free tier has limits)
- **Important**: Old `api-inference.huggingface.co` endpoints are deprecated (410 error)
- Use Inference Endpoints or router endpoints instead

### Server Won't Start
1. Check if Python is installed: `python --version`
2. Verify dependencies: `pip list`
3. Check `.env` file exists and has correct format
4. Look for error messages in the terminal

## 🔧 Development

### Running in Debug Mode
Debug mode is enabled by default when `FLASK_DEBUG=True` in `.env`. This provides:
- Auto-reload on code changes
- Detailed error messages
- Debug toolbar

### Testing the API
You can test the health endpoint:
```bash
curl http://localhost:5000/health
```

## 📦 Production Deployment

For production deployment, see `DEPLOYMENT.md` for detailed instructions on:
- Docker deployment
- Cloud platform deployment (Heroku, AWS, etc.)
- Using Gunicorn/WSGI servers
- Environment variable management

## 📝 License

This project is open source and available for personal and commercial use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests.

---

**Enjoy chatting with your AI assistant!** 🎉
