# API Keys Guide - Quick Start

## 🔑 Default API Keys (Ready to Use)

The following API keys are **already configured** and ready to use:

### Demo API Key
```
demo-api-key-12345
```
- **Rate Limit**: 100 requests/minute
- **Access**: Standard chat + Raw data

### Premium API Key
```
premium-api-key-67890
```
- **Rate Limit**: 1000 requests/minute
- **Access**: Standard chat + Raw data

## 📝 How to Use API Keys

### 1. Access Raw Data Endpoint

```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "What is 10+10?",
    "conversation_id": "test-123"
  }'
```

### 2. Standard Chat with API Key

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "Hello",
    "conversation_id": "test-123"
  }'
```

### 3. Check Your API Key Status

```bash
curl -X GET http://localhost:8000/api/auth/info \
  -H "X-API-Key: demo-api-key-12345"
```

## 🔧 Adding Your Own API Keys

### Option 1: Environment Variables (Recommended)

Create a `.env` file in your project root:

```env
API_KEY_1=your-custom-key-1
API_KEY_2=your-custom-key-2
DEFAULT_API_KEY=your-custom-key-1
```

Then update `main.py` to load from environment:

```python
API_KEYS = {
    os.getenv("API_KEY_1", "demo-api-key-12345"): {"name": "Custom Key 1", "rate_limit": 100},
    os.getenv("API_KEY_2", "premium-api-key-67890"): {"name": "Custom Key 2", "rate_limit": 1000},
}
```

### Option 2: Direct in main.py

Edit `main.py` and add your keys:

```python
API_KEYS = {
    "your-secret-key-here": {"name": "My Key", "rate_limit": 500},
    "another-key-here": {"name": "Team Key", "rate_limit": 2000},
    # Keep existing keys
    "demo-api-key-12345": {"name": "Demo Key", "rate_limit": 100},
    "premium-api-key-67890": {"name": "Premium Key", "rate_limit": 1000},
}
```

## 🧪 Testing Your API Key

### Test 1: Verify Authentication
```bash
curl -X GET http://localhost:8000/api/auth/info \
  -H "X-API-Key: demo-api-key-12345"
```

Expected response:
```json
{
  "authenticated": true,
  "rate_limit": 100,
  "api_key_name": "Demo Key"
}
```

### Test 2: Get Raw Data
```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{"message": "Hello", "conversation_id": "test"}'
```

## ⚠️ Important Notes

1. **Raw Data Requires Authentication**: The `/api/chat/raw` endpoint requires a valid API key
2. **Rate Limits**: Each API key has its own rate limit
3. **Security**: Never commit API keys to version control
4. **Production**: Use environment variables or a secure key management system

## 🚫 Without API Key

If you try to access raw data without an API key:

```bash
curl -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

You'll get:
```json
{
  "detail": "Authentication required for raw data access"
}
```

## 📊 Rate Limit Information

- **No API Key**: 10 requests/minute
- **Demo Key**: 100 requests/minute
- **Premium Key**: 1000 requests/minute
- **Custom Keys**: Configurable

## 🔐 Generate Secure API Keys

To generate a secure API key, you can use:

```python
import secrets
api_key = secrets.token_urlsafe(32)
print(f"Your API key: {api_key}")
```

Or use online tools like:
- https://randomkeygen.com/
- https://www.uuidgenerator.net/

## 📍 Where to Find Your API Keys

1. **Default Keys**: Already in `main.py` (lines 32-40)
2. **Environment Variables**: Check your `.env` file
3. **Custom Keys**: In `main.py` under `API_KEYS` dictionary

## 🎯 Quick Reference

| Endpoint | Requires API Key | Rate Limit |
|----------|-----------------|------------|
| `/api/chat` | Optional | 10/min (no key) or key-based |
| `/api/chat/raw` | **Required** | 20/min |
| `/api/auth/info` | Optional | No limit |
| `/` (Web UI) | No | No limit |

