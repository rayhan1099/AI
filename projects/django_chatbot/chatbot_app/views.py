from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import requests
import uuid
from .models import Conversation, Message

# Free AI API configuration
USE_OLLAMA = True
OLLAMA_URL = "http://localhost:11434/api/generate"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

def get_ai_response(message, conversation_history=None):
    """Get AI response using free API"""
    try:
        if USE_OLLAMA:
            # Use Ollama (local, free)
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
                return "Ollama is not running. Please start Ollama or use Hugging Face API."
        else:
            # Use Hugging Face API (free tier)
            headers = {"Authorization": "Bearer YOUR_FREE_API_KEY"}
            payload = {
                "inputs": message,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.7
                }
            }
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "Sorry, I couldn't generate a response.")
                return result.get("generated_text", "Sorry, I couldn't generate a response.")
            return "AI service is temporarily unavailable."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def index(request):
    """Render chat interface"""
    return render(request, 'chatbot/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """Handle chat messages"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        session_id = data.get('session_id')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get or create conversation
        if not session_id:
            session_id = str(uuid.uuid4())
        
        conversation, created = Conversation.objects.get_or_create(
            session_id=session_id,
            defaults={'user': request.user if request.user.is_authenticated else None}
        )
        
        # Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )
        
        # Get conversation history
        history = Message.objects.filter(conversation=conversation).order_by('created_at')
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history[:10]  # Last 10 messages for context
        ]
        
        # Get AI response
        ai_response = get_ai_response(message, conversation_history)
        
        # Save AI response
        ai_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=ai_response
        )
        
        return JsonResponse({
            'session_id': session_id,
            'user_message': message,
            'ai_response': ai_response,
            'message_id': ai_message.id
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def history(request, session_id):
    """Get conversation history"""
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        messages = Message.objects.filter(conversation=conversation).order_by('created_at')
        
        history = [
            {
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at.isoformat()
            }
            for msg in messages
        ]
        
        return JsonResponse({'history': history})
    
    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

