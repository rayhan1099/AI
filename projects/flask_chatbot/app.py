from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import uuid
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# CORS
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Free AI API configuration
USE_OLLAMA = True
OLLAMA_URL = "http://localhost:11434/api/generate"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HUGGINGFACE_API_KEY = "YOUR_FREE_API_KEY"

# In-memory conversation storage (use database in production)
conversations = {}

def get_ai_response(message, conversation_history=None):
    """Get AI response using free API"""
    try:
        if USE_OLLAMA:
            # Use Ollama (local, completely free)
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "llama2",
                    "prompt": message,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "Sorry, I couldn't generate a response.")
            else:
                return "Ollama is not running. Please start Ollama or set USE_OLLAMA=False"
        else:
            # Use Hugging Face Inference API (free tier)
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            payload = {
                "inputs": message,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            response = requests.post(
                HUGGINGFACE_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "Sorry, I couldn't generate a response.")
                return result.get("generated_text", "Sorry, I couldn't generate a response.")
            return "AI service is temporarily unavailable. Please try again later."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return "An error occurred. Please try again."

@app.route('/')
def index():
    """Render chat interface"""
    # Initialize session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Initialize conversation history
    if session['session_id'] not in conversations:
        conversations[session['session_id']] = []
    
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize conversation if needed
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message to history
        conversations[session_id].append({"role": "user", "content": message})
        
        # Get conversation history (last 10 messages for context)
        history = conversations[session_id][-10:]
        
        # Get AI response
        ai_response = get_ai_response(message, history)
        
        # Add AI response to history
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        return jsonify({
            'session_id': session_id,
            'user_message': message,
            'ai_response': ai_response
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        session_id = session.get('session_id')
        
        if not session_id or session_id not in conversations:
            return jsonify({'history': []})
        
        return jsonify({'history': conversations[session_id]})
    
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversations:
            conversations[session_id] = []
        return jsonify({'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Flask Chatbot'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

