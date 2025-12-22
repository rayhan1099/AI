# API Authentication & Rate Limiting Guide

## Overview
The FastAPI Chatbot now includes:
- **Rate Limiting**: Prevents API abuse
- **API Key Authentication**: Secure access control
- **Exact AI Response Data**: Returns raw API responses when requested

## API Key Authentication

### Default API Keys
- **Demo Key**: `demo-api-key-12345` (100 requests/minute)
- **Premium Key**: `premium-api-key-67890` (1000 requests/minute)

### Using API Keys

#### 1. Basic Chat Endpoint (with authentication)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "Hello, how are you?",
    "conversation_id": "test-123"
  }'
```

#### 2. Get Raw AI Response Data (requires authentication)
```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "What is 10+10?",
    "conversation_id": "test-123"
  }'
```

#### 3. Check Authentication Status
```bash
curl -X GET http://localhost:8000/api/auth/info \
  -H "X-API-Key: demo-api-key-12345"
```

## Rate Limiting

### Limits
- **Unauthenticated users**: 10 requests/minute
- **Authenticated users**: Based on API key tier (100-1000 requests/minute)
- **Raw data endpoint**: 20 requests/minute

### Rate Limit Headers
Responses include rate limit information:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets

### Rate Limit Exceeded
When rate limit is exceeded, you'll receive:
```json
{
  "detail": "Rate limit exceeded: 10 per 1 minute"
}
```
Status code: `429 Too Many Requests`

## Response Formats

### Standard Response
```json
{
  "user_message": "Hello",
  "conversation_id": "test-123",
  "ai_response": "Hello! I'm your AI assistant...",
  "success": true,
  "source": "ollama",
  "authenticated": true
}
```

### Raw Response (with return_raw=true and authentication)
```json
{
  "user_message": "Hello",
  "conversation_id": "test-123",
  "ai_response": "Hello! I'm your AI assistant...",
  "raw_api_response": {
    "response": "Hello! I'm your AI assistant...",
    "model": "llama2",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "metadata": {
    "model": "llama2",
    "status_code": 200,
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "success": true,
  "source": "ollama",
  "authenticated": true,
  "api_key_name": "Demo Key"
}
```

## Environment Variables

Set these in your environment or `.env` file:

```bash
# API Keys
API_KEY_1=demo-api-key-12345
API_KEY_2=premium-api-key-67890
DEFAULT_API_KEY=demo-api-key-12345

# Hugging Face (if not using Ollama)
HUGGINGFACE_API_KEY=your_huggingface_key_here
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables for production**
3. **Rotate API keys regularly**
4. **Monitor rate limit usage**
5. **Use HTTPS in production**

## Error Responses

### Invalid API Key (401)
```json
{
  "detail": "Invalid API key"
}
```

### Rate Limit Exceeded (429)
```json
{
  "detail": "Rate limit exceeded: 10 per 1 minute"
}
```

### Missing Message (400)
```json
{
  "detail": "Message is required"
}
```

## Web Interface

The web interface at `http://localhost:8000` does not require authentication and uses WebSocket connections with built-in rate limiting.

## Testing

### Test without API key (public access)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Test with API key
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{"message": "Hello"}'
```

### Test raw data endpoint
```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{"message": "What is 10+10?"}'
```

