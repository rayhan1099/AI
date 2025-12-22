from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import requests
import uuid
import json
import logging
import os
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB

# CORS
CORS(app)

# Rate limiting
rate_limit_per_day = os.getenv('RATE_LIMIT_PER_DAY', '200')
rate_limit_per_hour = os.getenv('RATE_LIMIT_PER_HOUR', '50')
rate_limit_per_minute = os.getenv('RATE_LIMIT_PER_MINUTE', '10')

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{rate_limit_per_day} per day", f"{rate_limit_per_hour} per hour"]
)

# AI API configuration from environment variables
USE_OLLAMA = os.getenv('USE_OLLAMA', 'False').lower() == 'true'
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
# Note: Hugging Face deprecated api-inference.huggingface.co
# You may need to use Inference Endpoints or a different service
# For now, we'll try the router endpoint, but you may need to configure
# your own Inference Endpoint or use an alternative service
HUGGINGFACE_API_URL = os.getenv('HUGGINGFACE_API_URL', 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium')
HUGGINGFACE_API_KEY = os.getenv('HF_TOKEN', '')
MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))

# Alternative models (uncomment to use):
# HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/gpt2'
# HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill'

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
                    "model": OLLAMA_MODEL,
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
            if not HUGGINGFACE_API_KEY:
                logger.error("Hugging Face API key is missing")
                return "Configuration error: Hugging Face token is not set. Please check your .env file."
            
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            
            # For DialoGPT, use simple message format
            # Build conversation context if available
            if conversation_history and len(conversation_history) > 0:
                # DialoGPT works better with conversation history
                past_responses = []
                for msg in conversation_history[-6:]:  # Last 6 messages
                    if msg.get("role") == "assistant":
                        past_responses.append(msg.get("content", ""))
                
                # Use the last assistant response as context
                if past_responses:
                    context = past_responses[-1] + " " + message
                else:
                    context = message
            else:
                context = message
            
            payload = {
                "inputs": context,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Retry logic for model loading (503 status)
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    HUGGINGFACE_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60  # Increased timeout for model loading
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    generated_text = ""
                    
                    if isinstance(result, list) and len(result) > 0:
                        # List format: [{"generated_text": "..."}]
                        if isinstance(result[0], dict):
                            generated_text = result[0].get("generated_text", "")
                        else:
                            generated_text = str(result[0])
                    elif isinstance(result, dict):
                        # Dict format: {"generated_text": "..."}
                        generated_text = result.get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Clean up the response
                    if generated_text:
                        # Remove the input context if it's included in the response
                        if context and context in generated_text:
                            generated_text = generated_text.replace(context, "").strip()
                        
                        # Clean up common artifacts
                        generated_text = generated_text.strip()
                        
                        # Take the first meaningful sentence/response
                        # Remove any remaining context markers
                        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
                        if lines:
                            response_text = lines[0]
                            # Remove common prefixes
                            for prefix in ["User:", "Assistant:", "Bot:", "AI:"]:
                                if response_text.startswith(prefix):
                                    response_text = response_text[len(prefix):].strip()
                            
                            if response_text and len(response_text) > 0:
                                return response_text
                    
                    return "Sorry, I couldn't generate a proper response. Please try again."
                
                elif response.status_code == 410:
                    # Endpoint deprecated - try router endpoint
                    logger.warning("Old API endpoint deprecated, trying router endpoint")
                    router_url = HUGGINGFACE_API_URL.replace("api-inference.huggingface.co", "router.huggingface.co")
                    router_url = router_url.replace("/models/", "/")
                    
                    try:
                        router_response = requests.post(
                            router_url,
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        if router_response.status_code == 200:
                            result = router_response.json()
                            # Process result same as before
                            if isinstance(result, list) and len(result) > 0:
                                generated_text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
                            elif isinstance(result, dict):
                                generated_text = result.get("generated_text", "")
                            else:
                                generated_text = str(result)
                            
                            if generated_text and generated_text.strip():
                                return generated_text.strip()
                        elif router_response.status_code == 503:
                            return "The AI model is loading. Please wait 30-60 seconds and try again."
                    except:
                        pass
                    
                    return "⚠️ Hugging Face API endpoint has changed. Please update HUGGINGFACE_API_URL in .env to use Inference Endpoints or router.huggingface.co. See README for details."
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        logger.info(f"Model is loading, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "The AI model is currently loading. This can take 30-60 seconds. Please wait a moment and try again."
                
                elif response.status_code == 401:
                    logger.error("Unauthorized: Invalid Hugging Face token")
                    return "Authentication error: Please check your Hugging Face token in the .env file."
                
                elif response.status_code == 429:
                    logger.error("Rate limit exceeded")
                    return "Rate limit exceeded. Please wait a moment before sending another message."
                
                else:
                    # Log the error for debugging
                    error_detail = response.text[:200] if response.text else "No error details"
                    logger.error(f"Hugging Face API error: Status {response.status_code}, Response: {error_detail}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return f"AI service error (Status {response.status_code}). Please try again in a moment."
            
            return "AI service is temporarily unavailable. Please try again later."
            
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return "Request timed out. The AI service might be slow. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return "Connection error. Please check your internet connection and try again."
    except Exception as e:
        logger.error(f"AI response error: {e}", exc_info=True)
        return f"An error occurred: {str(e)}. Please try again."

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
@limiter.limit(f"{rate_limit_per_minute} per minute")
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
        
        # Get conversation history (last N messages for context)
        history = conversations[session_id][-MAX_CONVERSATION_HISTORY:]
        
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
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host=host, port=port, debug=debug)

