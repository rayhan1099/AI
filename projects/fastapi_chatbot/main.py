from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import requests
import logging
from typing import List
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chatbot", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Free AI API configuration (Hugging Face Inference API)
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HUGGINGFACE_API_KEY = "YOUR_FREE_API_KEY"  # Get from huggingface.co

# Alternative: Use local model with Ollama
USE_OLLAMA = True  # Set to False to use Hugging Face API
OLLAMA_URL = "http://localhost:11434/api/generate"

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

async def get_ai_response(message: str, conversation_id: str) -> str:
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
            else:
                return "AI service is temporarily unavailable. Please try again later."
    
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return "An error occurred. Please try again."

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
            ai_response = await get_ai_response(user_message, client_id)
            
            # Add AI response to history
            conversations[client_id].append({"role": "assistant", "content": ai_response})
            
            # Send AI response to client
            await manager.send_personal_message(
                json.dumps({
                    "type": "ai_message",
                    "message": ai_response
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
    return {"status": "healthy", "service": "FastAPI Chatbot"}

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """REST API endpoint for chat"""
    data = await request.json()
    message = data.get("message", "")
    conversation_id = data.get("conversation_id", "default")
    
    if not message:
        return {"error": "Message is required"}
    
    # Get AI response
    ai_response = await get_ai_response(message, conversation_id)
    
    return {
        "user_message": message,
        "ai_response": ai_response,
        "conversation_id": conversation_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

