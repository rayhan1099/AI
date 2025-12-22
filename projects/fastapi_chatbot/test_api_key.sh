#!/bin/bash
# Quick test script for API key authentication

echo "=== Testing API Key Authentication ==="
echo ""

# Test 1: Check auth info
echo "1. Testing /api/auth/info endpoint..."
curl -s -X GET http://localhost:8000/api/auth/info \
  -H "X-API-Key: demo-api-key-12345" | python -m json.tool
echo ""
echo ""

# Test 2: Get raw data
echo "2. Testing /api/chat/raw endpoint with API key..."
curl -s -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "message": "What is 10+10?",
    "conversation_id": "test-123"
  }' | python -m json.tool
echo ""
echo ""

# Test 3: Try without API key (should fail)
echo "3. Testing /api/chat/raw WITHOUT API key (should fail)..."
curl -s -X POST http://localhost:8000/api/chat/raw \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "conversation_id": "test-123"
  }' | python -m json.tool
echo ""
echo ""

echo "=== Tests Complete ==="
echo ""
echo "Default API Keys you can use:"
echo "  - demo-api-key-12345 (100 req/min)"
echo "  - premium-api-key-67890 (1000 req/min)"

