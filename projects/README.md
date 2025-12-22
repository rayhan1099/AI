# Complete AI Chatbot Projects

Three complete chatbot projects using FastAPI, Django, and Flask with modern UI and free AI integration.

## 📁 Projects

### 1. FastAPI Chatbot
**Location**: `fastapi_chatbot/`

**Features**:
- WebSocket support for real-time chat
- Modern gradient UI (purple theme)
- Free AI integration (Ollama/Hugging Face)
- REST API fallback

**Quick Start**:
```bash
cd fastapi_chatbot
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Django Chatbot
**Location**: `django_chatbot/`

**Features**:
- Database integration (SQLite)
- Admin panel
- Conversation history
- Modern gradient UI (purple theme)

**Quick Start**:
```bash
cd django_chatbot
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 3. Flask Chatbot
**Location**: `flask_chatbot/`

**Features**:
- Simple and lightweight
- Session-based conversations
- Clear history button
- Modern gradient UI (pink theme)

**Quick Start**:
```bash
cd flask_chatbot
pip install -r requirements.txt
python app.py
```

## 🤖 Free AI Setup

### Option 1: Ollama (Recommended - Completely Free)

```bash
# Install Ollama
# Visit: https://ollama.ai

# Pull a model
ollama pull llama2

# The chatbot will automatically use Ollama if running
```

### Option 2: Hugging Face (Free Tier)

1. Sign up at https://huggingface.co
2. Get your free API key
3. Update `HUGGINGFACE_API_KEY` in the code
4. Set `USE_OLLAMA = False`

## 🎨 UI Features

All three projects include:
- Modern gradient design
- Responsive layout
- Real-time typing indicators
- Smooth animations
- Mobile-friendly
- Auto-scrolling chat
- Message history

## 🔧 Configuration

Each project has:
- `USE_OLLAMA = True` - Use local Ollama (default)
- `USE_OLLAMA = False` - Use Hugging Face API

## 📝 Notes

- All projects use free AI APIs
- No credit card required
- Works offline with Ollama
- Production-ready structure
- Complete error handling

## 🚀 Deployment

Each project includes:
- Docker support (add Dockerfile)
- Requirements.txt
- Production-ready code
- Error handling
- Rate limiting (Flask/Django)

---

**Choose your framework and start building!**

