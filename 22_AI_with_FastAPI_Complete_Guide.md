# AI Development with FastAPI - Complete Guide

## 📖 Table of Contents
1. [FastAPI Setup for AI](#fastapi-setup-for-ai)
2. [AI Models Integration](#ai-models-integration)
3. [Building AI APIs](#building-ai-apis)
4. [Advanced Features](#advanced-features)
5. [Complete Project Examples](#complete-project-examples)
6. [Best Practices](#best-practices)
7. [Deployment](#deployment)

---

## FastAPI Setup for AI

### Project Structure

```
ai_fastapi_project/
├── main.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── ml_models.py       # AI models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── sentiment.py
│   │   ├── classification.py
│   │   └── image.py
│   └── utils.py
├── ml_models/
│   └── trained_models/
└── tests/
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install FastAPI and AI libraries
pip install fastapi uvicorn[standard]
pip install transformers torch scikit-learn joblib
pip install python-multipart  # For file uploads
pip install aiofiles  # Async file operations
pip install redis  # For caching
```

### requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
transformers==4.35.0
torch==2.1.0
scikit-learn==1.3.2
joblib==1.3.2
python-multipart==0.0.6
aiofiles==23.2.1
redis==5.0.1
Pillow==10.1.0
numpy==1.24.3
pandas==2.1.3
```

### Basic FastAPI App

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI API",
    description="FastAPI application with AI capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI API is running", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

## AI Models Integration

### ML Models Manager

```python
# app/ml_models.py
import os
import joblib
import torch
from transformers import pipeline
from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AIModelManager:
    """Centralized AI model manager with caching"""
    
    _models = {}
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_sentiment_model(cls):
        """Lazy load sentiment analysis model"""
        if 'sentiment' not in cls._models:
            logger.info("Loading sentiment model...")
            cls._models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Sentiment model loaded")
        return cls._models['sentiment']
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_text_classifier(cls):
        """Load custom text classifier"""
        if 'text_classifier' not in cls._models:
            model_path = "ml_models/trained_models/text_classifier.pkl"
            if os.path.exists(model_path):
                logger.info("Loading text classifier...")
                cls._models['text_classifier'] = joblib.load(model_path)
                logger.info("Text classifier loaded")
        return cls._models.get('text_classifier')
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_summarizer(cls):
        """Load text summarization model"""
        if 'summarizer' not in cls._models:
            logger.info("Loading summarizer...")
            cls._models['summarizer'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            logger.info("Summarizer loaded")
        return cls._models['summarizer']
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_image_classifier(cls):
        """Load image classification model"""
        if 'image_classifier' not in cls._models:
            from torchvision import models, transforms
            logger.info("Loading image classifier...")
            cls._models['image_classifier'] = models.resnet50(pretrained=True)
            cls._models['image_classifier'].eval()
            logger.info("Image classifier loaded")
        return cls._models['image_classifier']

# Global instance
model_manager = AIModelManager()
```

---

## Building AI APIs

### Project 1: Sentiment Analysis API

```python
# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

# app/routers/sentiment.py
from fastapi import APIRouter, HTTPException
from app.models import TextRequest, SentimentResponse, BatchTextRequest, BatchSentimentResponse
from app.ml_models import model_manager

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    """Analyze sentiment of a single text"""
    try:
        model = model_manager.get_sentiment_model()
        result = model(request.text)[0]
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['label'].lower(),
            confidence=result['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchTextRequest):
    """Analyze sentiment of multiple texts"""
    try:
        model = model_manager.get_sentiment_model()
        results = []
        
        for text in request.texts:
            result = model(text)[0]
            results.append(SentimentResponse(
                text=text,
                sentiment=result['label'].lower(),
                confidence=result['score']
            ))
        
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Include Routers

```python
# main.py
from app.routers import sentiment, classification, image

app.include_router(sentiment.router)
app.include_router(classification.router)
app.include_router(image.router)
```

---

## Advanced Features

### Project 2: Text Summarization with Async

```python
# app/routers/summarization.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from app.ml_models import model_manager
import asyncio

router = APIRouter(prefix="/summarize", tags=["summarization"])

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=100)
    max_length: int = Field(130, ge=30, le=200)
    min_length: int = Field(30, ge=10, le=100)

class SummarizeResponse(BaseModel):
    original_text: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

@router.post("/", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text asynchronously"""
    try:
        # Run model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        summarizer = model_manager.get_summarizer()
        
        # Run blocking operation in thread pool
        result = await loop.run_in_executor(
            None,
            lambda: summarizer(
                request.text,
                max_length=request.max_length,
                min_length=request.min_length,
                do_sample=False
            )
        )
        
        summary = result[0]['summary_text']
        compression_ratio = len(summary) / len(request.text)
        
        return SummarizeResponse(
            original_text=request.text,
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            compression_ratio=compression_ratio
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Project 3: Image Classification

```python
# app/routers/image.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
from app.ml_models import model_manager
from typing import List

router = APIRouter(prefix="/image", tags=["image"])

class Prediction(BaseModel):
    class_name: str
    confidence: float

class ImageClassificationResponse(BaseModel):
    filename: str
    predictions: List[Prediction]

@router.post("/classify", response_model=ImageClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """Classify uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get model
        model = model_manager.get_image_classifier()
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict (run in thread pool)
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            outputs = await loop.run_in_executor(
                None,
                lambda: model(image_tensor)
            )
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Load ImageNet classes
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Format results
        predictions = [
            Prediction(
                class_name=classes[idx],
                confidence=float(prob)
            )
            for prob, idx in zip(top5_prob, top5_idx)
        ]
        
        return ImageClassificationResponse(
            filename=file.filename,
            predictions=predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Complete Project Examples

### Project 4: Complete AI Service with Database

```python
# app/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_predictions.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    prediction_type = Column(String)
    result = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# app/dependencies.py
from app.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import Prediction, get_db
from app.models import TextRequest, SentimentResponse
from app.ml_models import model_manager

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/sentiment", response_model=SentimentResponse)
async def predict_and_save(
    request: TextRequest,
    db: Session = Depends(get_db)
):
    """Predict sentiment and save to database"""
    try:
        model = model_manager.get_sentiment_model()
        result = model(request.text)[0]
        
        # Save to database
        prediction = Prediction(
            text=request.text,
            prediction_type="sentiment",
            result=result['label'],
            confidence=result['score']
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['label'].lower(),
            confidence=result['score']
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_prediction_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    predictions = db.query(Prediction).offset(skip).limit(limit).all()
    return predictions
```

### Project 5: AI Chatbot with Streaming

```python
# app/routers/chatbot.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    response: str

# Load model (do this once)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat with AI"""
    try:
        # Encode conversation
        conversation = message.conversation_history + [message.message]
        full_text = " ".join(conversation)
        
        # Tokenize
        inputs = tokenizer.encode(full_text + tokenizer.eos_token, return_tensors='pt')
        
        # Generate (in thread pool)
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: model.generate(
                inputs,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(full_text):].strip()
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(message: ChatMessage):
    """Stream chat response"""
    async def generate():
        # Your streaming logic here
        for chunk in response_chunks:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Best Practices

### 1. Error Handling Middleware

```python
# main.py
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
```

### 2. Rate Limiting

```python
# app/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 10, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip]
            if now - timestamp < self.period
        ]
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Add current request
        self.clients[client_ip].append(now)
        
        response = await call_next(request)
        return response

# Use in main.py
app.add_middleware(RateLimitMiddleware, calls=10, period=60)
```

### 3. Caching

```python
# app/cache.py
from functools import wraps
import hashlib
import json
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
            cache_key = f"{func.__name__}:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@router.post("/analyze")
@cache_result(expiration=3600)
async def analyze_with_cache(request: TextRequest):
    # Your code
    pass
```

### 4. Logging

```python
# main.py
import logging
from fastapi import Request
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}s"
    )
    
    return response
```

---

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./ml_models:/app/ml_models
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Production Settings

```python
# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

---

## Key Takeaways

1. **FastAPI is perfect for AI** - Fast, async, auto-docs
2. **Use routers** - Organize endpoints
3. **Async operations** - Use thread pools for blocking AI
4. **Pydantic models** - Type safety and validation
5. **Caching** - Cache model results
6. **Error handling** - Proper exception handling
7. **Database integration** - Store predictions

---

**Master FastAPI AI development with these complete examples!**

