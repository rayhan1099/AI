from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import json
import requests
import logging
from typing import List, Optional
import asyncio
import os
from datetime import datetime, timedelta
import secrets

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("✅ Loaded environment variables from .env file")
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OpenAI for Hugging Face Inference Providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Install with: pip install openai")

app = FastAPI(title="AI Chatbot", version="1.0.0")

# Rate Limiting Setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key Authentication
# In production, store these in environment variables or a database
API_KEYS = {
    os.getenv("API_KEY_1", "demo-api-key-12345"): {"name": "Demo Key", "rate_limit": 100},
    os.getenv("API_KEY_2", "premium-api-key-67890"): {"name": "Premium Key", "rate_limit": 1000},
}

# Generate a default API key if none exists
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "demo-api-key-12345")
if DEFAULT_API_KEY not in API_KEYS:
    API_KEYS[DEFAULT_API_KEY] = {"name": "Default Key", "rate_limit": 100}

# Security
security = HTTPBearer(auto_error=False)

def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> dict:
    """Verify API key and return user info"""
    if not api_key:
        # Allow public access to web interface, but require API key for API endpoints
        return {"authenticated": False, "rate_limit": 10}
    
    if api_key in API_KEYS:
        return {
            "authenticated": True,
            "api_key": api_key,
            "rate_limit": API_KEYS[api_key]["rate_limit"],
            "name": API_KEYS[api_key]["name"]
        }
    
    raise HTTPException(status_code=401, detail="Invalid API key")

def get_rate_limit(request: Request) -> str:
    """Get rate limit based on API key"""
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key in API_KEYS:
        limit = API_KEYS[api_key].get("rate_limit", 100)
        return f"{limit}/minute"
    return "10/minute"  # Default rate limit for unauthenticated users

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Hugging Face Inference Providers (OpenAI-compatible API)
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY", "YOUR_HUGGINGFACE_API_KEY"))
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"  # OpenAI-compatible endpoint
HUGGINGFACE_MODEL = os.getenv("HF_MODEL", "moonshotai/Kimi-K2-Instruct-0905")  # Default model

# Alternative: Use local model with Ollama
# Set to True only if Ollama is installed and running
USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() == "true"  # Default to False
OLLAMA_URL = "http://localhost:11434/api/generate"

# Track if we've warned about Ollama connection
_ollama_warning_logged = False

def check_ollama_availability():
    """Check if Ollama is running (non-blocking check)"""
    global _ollama_warning_logged
    if USE_OLLAMA and not _ollama_warning_logged:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code == 200:
                logger.info("✅ Ollama is running and available")
                return True
        except:
            if not _ollama_warning_logged:
                logger.warning("⚠️  Ollama is not running. Using fallback responses. To use Ollama: 1) Install Ollama, 2) Start it, 3) Set USE_OLLAMA=True")
                _ollama_warning_logged = True
            return False
    return False

class ConnectionManager:
    """Manage WebSocket connections"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Conversation history storage (in production, use database)
conversations = {}

def safe_eval_math(expression: str):
    """Safely evaluate mathematical expressions"""
    import re
    # Extract math expression from text (numbers and operators)
    # First, try to find a math expression pattern
    math_expr = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', expression)
    if math_expr:
        num1 = float(math_expr.group(1))
        op = math_expr.group(2)
        num2 = float(math_expr.group(3))
        
        try:
            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                if num2 == 0:
                    return False, None
                result = num1 / num2
            else:
                return False, None
            
            # Return as int if it's a whole number, otherwise float
            if result == int(result):
                return True, int(result)
            return True, result
        except:
            return False, None
    
    # Fallback: try to clean and evaluate if it's a pure math expression
    cleaned = re.sub(r'[^0-9+\-*/().\s]', '', expression).strip()
    if cleaned and re.match(r'^[0-9+\-*/().\s]+$', cleaned):
        try:
            result = eval(cleaned, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                if result == int(result):
                    return True, int(result)
                return True, result
        except:
            pass
    
    return False, None

def get_fallback_response(message: str) -> str:
    """Simple fallback responses when AI services are unavailable"""
    try:
        if not message:
            message = "hello"
        
        message_lower = message.lower().strip()
        
        # Mathematical calculations - check first before other patterns
        import re
        
        # Handle word-based math operations
        word_math_patterns = [
            (r'(\d+)\s+plus\s+(\d+)', lambda a, b: int(a) + int(b)),
            (r'(\d+)\s+minus\s+(\d+)', lambda a, b: int(a) - int(b)),
            (r'(\d+)\s+times\s+(\d+)', lambda a, b: int(a) * int(b)),
            (r'(\d+)\s+multiplied\s+by\s+(\d+)', lambda a, b: int(a) * int(b)),
            (r'(\d+)\s+divided\s+by\s+(\d+)', lambda a, b: int(a) / int(b) if int(b) != 0 else None),
        ]
        
        for pattern, operation in word_math_patterns:
            match = re.search(pattern, message_lower)
            if match:
                try:
                    num1 = int(match.group(1))
                    num2 = int(match.group(2))
                    result = operation(num1, num2)
                    if result is not None:
                        if result == int(result):
                            return f"The answer is **{int(result)}**."
                        return f"The answer is **{result}**."
                except:
                    pass
        
        # Try to extract and calculate math expressions
        success, result = safe_eval_math(message)
        if success:
            return f"The answer is **{result}**."
        
        # Greetings
        if any(word in message_lower for word in ["hi", "hello", "hey", "greetings"]):
            return "Hello! I'm your AI assistant. How can I help you today?"
        
        # Questions about calculations
        if any(word in message_lower for word in ["calculate", "compute", "solve", "math", "arithmetic"]):
            return "I can help with basic math! Try asking something like 'what is 10+10' or '5*3'."
        
        # Questions
        if "?" in message:
            if "how" in message_lower:
                return "That's a great question! I'm here to help, though I'm currently running in fallback mode. Could you provide more details?"
            elif "what" in message_lower:
                # Check if it's a math question
                if any(op in message_lower for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]):
                    success, result = safe_eval_math(message)
                    if success:
                        return f"The answer is **{result}**."
                return "I'd be happy to explain! However, I'm currently using a simple response system. Could you rephrase your question?"
            else:
                return "Interesting question! I'm currently in fallback mode, so I may not have the full context. Could you elaborate?"
        
        # Error mentions
        if any(word in message_lower for word in ["error", "problem", "issue", "wrong", "broken"]):
            return "I understand you're experiencing an issue. The AI service isn't currently available, so I'm using a fallback response system. To enable full AI responses, please set up Ollama or configure a Hugging Face API key."
        
        # Default responses
        responses = [
            "I see! That's interesting. I'm currently running in fallback mode.",
            "Thanks for your message! I'm here to help, though I'm using a simple response system right now.",
            "Got it! I understand what you're saying. Currently, I'm operating in fallback mode.",
            "I hear you! To get more intelligent responses, please configure Ollama or a Hugging Face API key.",
        ]
        
        # Simple hash-based selection for variety
        index = abs(hash(message)) % len(responses)
        return responses[index]
    except Exception as e:
        logger.error(f"Fallback response error: {e}")
        return "Hello! I'm your AI assistant. How can I help you today?"

async def get_ai_response(message: str, conversation_id: str, return_raw: bool = False) -> dict:
    """Get AI response using free API - returns exact API response"""
    try:
        if USE_OLLAMA:
            # Use Ollama (local, completely free)
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": "llama2",
                        "prompt": message,
                        "stream": False
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    raw_data = response.json()
                    if return_raw:
                        return {
                            "success": True,
                            "raw_response": raw_data,
                            "response": raw_data.get("response", ""),
                            "source": "ollama",
                            "metadata": {
                                "model": "llama2",
                                "status_code": response.status_code,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    return {
                        "success": True,
                        "response": raw_data.get("response", get_fallback_response(message)),
                        "source": "ollama",
                        "raw_data": raw_data
                    }
                else:
                    logger.warning(f"Ollama returned status {response.status_code}")
                    return {
                        "success": False,
                        "response": get_fallback_response(message),
                        "source": "fallback",
                        "error": f"Ollama returned status {response.status_code}"
                    }
            except Exception as e:
                # Only log warning once to avoid spam
                check_ollama_availability()  # This logs warning once if Ollama is not running
                return {
                    "success": False,
                    "response": get_fallback_response(message),
                    "source": "fallback",
                    "error": "Ollama is not running"
                }
        else:
            # Use Hugging Face Inference Providers (OpenAI-compatible API)
            # Debug logging
            logger.info(f"Checking Hugging Face API key: {'SET' if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != 'YOUR_FREE_API_KEY' else 'NOT SET'}")
            logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
            
            if HUGGINGFACE_API_KEY == "YOUR_FREE_API_KEY" or not HUGGINGFACE_API_KEY:
                logger.warning(f"Hugging Face API key not configured. Current value: {HUGGINGFACE_API_KEY[:20] if HUGGINGFACE_API_KEY else 'None'}...")
                return {
                    "success": False,
                    "response": get_fallback_response(message),
                    "source": "fallback",
                    "error": "Hugging Face API key not configured. Set HF_TOKEN environment variable."
                }
            
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
                return {
                    "success": False,
                    "response": get_fallback_response(message),
                    "source": "fallback",
                    "error": "OpenAI package required for Hugging Face Inference Providers"
                }
            
            try:
                # Use OpenAI-compatible client for Hugging Face Inference Providers
                client = OpenAI(
                    base_url=HUGGINGFACE_BASE_URL,
                    api_key=HUGGINGFACE_API_KEY,
                )
                
                # Make chat completion request
                completion = client.chat.completions.create(
                    model=HUGGINGFACE_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": message
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Extract response
                ai_response_text = completion.choices[0].message.content
                
                if return_raw:
                    return {
                        "success": True,
                        "raw_response": {
                            "id": completion.id,
                            "model": completion.model,
                            "created": completion.created,
                            "choices": [
                                {
                                    "index": choice.index,
                                    "message": {
                                        "role": choice.message.role,
                                        "content": choice.message.content
                                    },
                                    "finish_reason": choice.finish_reason
                                } for choice in completion.choices
                            ],
                            "usage": {
                                "prompt_tokens": completion.usage.prompt_tokens,
                                "completion_tokens": completion.usage.completion_tokens,
                                "total_tokens": completion.usage.total_tokens
                            } if completion.usage else None
                        },
                        "response": ai_response_text,
                        "source": "huggingface_inference_providers",
                        "metadata": {
                            "model": completion.model,
                            "provider": getattr(completion, 'provider', 'unknown'),
                            "timestamp": datetime.now().isoformat(),
                            "tokens_used": completion.usage.total_tokens if completion.usage else None
                        }
                    }
                
                return {
                    "success": True,
                    "response": ai_response_text,
                    "source": "huggingface_inference_providers",
                    "raw_data": {
                        "model": completion.model,
                        "tokens": completion.usage.total_tokens if completion.usage else None
                    }
                }
                
            except Exception as e:
                logger.error(f"Hugging Face Inference Providers error: {e}")
                return {
                    "success": False,
                    "response": get_fallback_response(message),
                    "source": "fallback",
                    "error": str(e)
                }
    
    except requests.exceptions.Timeout:
        logger.warning("Request timed out")
        return {
            "success": False,
            "response": get_fallback_response(message),
            "source": "fallback",
            "error": "Request timed out"
        }
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return {
            "success": False,
            "response": get_fallback_response(message),
            "source": "fallback",
            "error": str(e)
        }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    
    # Initialize conversation history
    if client_id not in conversations:
        conversations[client_id] = []
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Add user message to history
            conversations[client_id].append({"role": "user", "content": user_message})
            
            # Send user message back to client (echo)
            await manager.send_personal_message(
                json.dumps({
                    "type": "user_message",
                    "message": user_message
                }),
                websocket
            )
            
            # Get AI response
            await manager.send_personal_message(
                json.dumps({
                    "type": "typing",
                    "status": True
                }),
                websocket
            )
            
            # Generate AI response
            ai_response_data = await get_ai_response(user_message, client_id, return_raw=False)
            ai_response = ai_response_data.get("response", "Sorry, I couldn't generate a response.")
            
            # Add AI response to history with metadata
            conversations[client_id].append({
                "role": "assistant", 
                "content": ai_response,
                "source": ai_response_data.get("source", "unknown"),
                "success": ai_response_data.get("success", False),
                "timestamp": datetime.now().isoformat()
            })
            
            # Send AI response to client with metadata
            await manager.send_personal_message(
                json.dumps({
                    "type": "ai_message",
                    "message": ai_response,
                    "source": ai_response_data.get("source", "unknown"),
                    "success": ai_response_data.get("success", False)
                }),
                websocket
            )
            
            await manager.send_personal_message(
                json.dumps({
                    "type": "typing",
                    "status": False
                }),
                websocket
            )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FastAPI Chatbot", "version": "1.0.0"}

@app.get("/api/auth/info")
async def auth_info(auth: dict = Depends(verify_api_key)):
    """Get authentication information"""
    return {
        "authenticated": auth.get("authenticated", False),
        "rate_limit": auth.get("rate_limit", 10),
        "api_key_name": auth.get("name", "Public Access") if auth.get("authenticated") else None
    }

@app.post("/api/chat")
@limiter.limit(get_rate_limit)
async def chat_endpoint(
    request: Request,
    auth: dict = Depends(verify_api_key)
):
    """REST API endpoint for chat with authentication and rate limiting"""
    try:
        data = await request.json()
        message = data.get("message", "")
        conversation_id = data.get("conversation_id", "default")
        return_raw = data.get("return_raw", False)
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get AI response with exact data
        ai_response_data = await get_ai_response(message, conversation_id, return_raw=return_raw)
        
        response = {
            "user_message": message,
            "conversation_id": conversation_id,
            "ai_response": ai_response_data.get("response", ""),
            "success": ai_response_data.get("success", False),
            "source": ai_response_data.get("source", "unknown"),
            "authenticated": auth.get("authenticated", False)
        }
        
        # Include raw data if requested and authenticated
        if return_raw and auth.get("authenticated"):
            response["raw_api_response"] = ai_response_data.get("raw_response")
            response["metadata"] = ai_response_data.get("metadata")
        elif return_raw:
            response["error"] = "Authentication required to access raw API responses"
        
        if ai_response_data.get("error"):
            response["error"] = ai_response_data.get("error")
        
        return response
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/raw")
@limiter.limit("20/minute")  # Rate limit for raw data endpoint
async def chat_raw_endpoint(
    request: Request,
    auth: dict = Depends(verify_api_key)
):
    """REST API endpoint for chat with raw AI response data (requires authentication)"""
    if not auth.get("authenticated"):
        raise HTTPException(status_code=401, detail="Authentication required for raw data access")
    
    try:
        data = await request.json()
        message = data.get("message", "")
        conversation_id = data.get("conversation_id", "default")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get AI response with raw data
        ai_response_data = await get_ai_response(message, conversation_id, return_raw=True)
        
        return {
            "user_message": message,
            "conversation_id": conversation_id,
            "ai_response": ai_response_data.get("response", ""),
            "raw_api_response": ai_response_data.get("raw_response"),
            "metadata": ai_response_data.get("metadata"),
            "success": ai_response_data.get("success", False),
            "source": ai_response_data.get("source", "unknown"),
            "authenticated": True,
            "api_key_name": auth.get("name", "Unknown")
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Chat raw endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

