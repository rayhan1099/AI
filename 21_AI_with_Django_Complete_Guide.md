# AI Development with Django - Complete Guide

## 📖 Table of Contents
1. [Django Setup for AI](#django-setup-for-ai)
2. [AI Models Integration](#ai-models-integration)
3. [Building AI APIs](#building-ai-apis)
4. [AI-Powered Features](#ai-powered-features)
5. [Complete Project Examples](#complete-project-examples)
6. [Best Practices](#best-practices)
7. [Deployment](#deployment)

---

## Django Setup for AI

### Project Structure

```
ai_django_project/
├── manage.py
├── requirements.txt
├── ai_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── ai_app/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── serializers.py
│   ├── ml_models.py
│   └── tasks.py
└── static/
└── templates/
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Django and AI libraries
pip install django djangorestframework
pip install transformers torch scikit-learn joblib
pip install celery redis  # For async tasks
pip install django-cors-headers
```

### requirements.txt

```txt
Django==4.2.7
djangorestframework==3.14.0
transformers==4.35.0
torch==2.1.0
scikit-learn==1.3.2
joblib==1.3.2
celery==5.3.4
redis==5.0.1
django-cors-headers==4.3.1
Pillow==10.1.0
numpy==1.24.3
pandas==2.1.3
```

### Settings Configuration

```python
# ai_project/settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'ai_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
]

# CORS settings for API
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
]

# Media files for AI model storage
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# ML Models directory
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
```

---

## AI Models Integration

### ML Models Manager

```python
# ai_app/ml_models.py
import os
import joblib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings
import numpy as np

class AIModelManager:
    """Centralized AI model manager for Django"""
    
    _sentiment_model = None
    _text_classifier = None
    _image_classifier = None
    
    @classmethod
    def get_sentiment_model(cls):
        """Lazy load sentiment analysis model"""
        if cls._sentiment_model is None:
            cls._sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        return cls._sentiment_model
    
    @classmethod
    def get_text_classifier(cls):
        """Load custom text classifier"""
        if cls._text_classifier is None:
            model_path = os.path.join(settings.ML_MODELS_DIR, 'text_classifier.pkl')
            if os.path.exists(model_path):
                cls._text_classifier = joblib.load(model_path)
        return cls._text_classifier
    
    @classmethod
    def get_image_classifier(cls):
        """Load image classification model"""
        if cls._image_classifier is None:
            from torchvision import models, transforms
            cls._image_classifier = models.resnet50(pretrained=True)
            cls._image_classifier.eval()
        return cls._image_classifier

# Usage
model_manager = AIModelManager()
```

---

## Building AI APIs

### Project 1: Sentiment Analysis API

```python
# ai_app/views.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .ml_models import AIModelManager
import json

@api_view(['POST'])
@permission_classes([AllowAny])
def analyze_sentiment(request):
    """Analyze sentiment of text"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return Response(
                {'error': 'Text is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get model
        model = AIModelManager.get_sentiment_model()
        
        # Analyze
        result = model(text)[0]
        
        return Response({
            'text': text,
            'sentiment': result['label'].lower(),
            'confidence': result['score']
        })
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([AllowAny])
def analyze_sentiment_batch(request):
    """Analyze multiple texts"""
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        
        if not texts:
            return Response(
                {'error': 'Texts array is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        model = AIModelManager.get_sentiment_model()
        results = []
        
        for text in texts:
            result = model(text)[0]
            results.append({
                'text': text,
                'sentiment': result['label'].lower(),
                'confidence': result['score']
            })
        
        return Response({'results': results})
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

### URLs Configuration

```python
# ai_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('sentiment/', views.analyze_sentiment, name='sentiment'),
    path('sentiment/batch/', views.analyze_sentiment_batch, name='sentiment_batch'),
]

# ai_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/ai/', include('ai_app.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

## AI-Powered Features

### Project 2: Text Classification with Database

```python
# ai_app/models.py
from django.db import models

class TextClassification(models.Model):
    text = models.TextField()
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

# ai_app/serializers.py
from rest_framework import serializers
from .models import TextClassification

class TextClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextClassification
        fields = ['id', 'text', 'predicted_class', 'confidence', 'created_at']
        read_only_fields = ['id', 'created_at']

# ai_app/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import TextClassification
from .serializers import TextClassificationSerializer
from .ml_models import AIModelManager

class TextClassificationViewSet(viewsets.ModelViewSet):
    queryset = TextClassification.objects.all()
    serializer_class = TextClassificationSerializer
    
    @action(detail=False, methods=['post'])
    def classify(self, request):
        """Classify text and save to database"""
        text = request.data.get('text', '')
        
        if not text:
            return Response(
                {'error': 'Text is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get model and classify
        model = AIModelManager.get_sentiment_model()
        result = model(text)[0]
        
        # Save to database
        classification = TextClassification.objects.create(
            text=text,
            predicted_class=result['label'],
            confidence=result['score']
        )
        
        serializer = self.get_serializer(classification)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get classification statistics"""
        total = TextClassification.objects.count()
        positive = TextClassification.objects.filter(
            predicted_class__icontains='positive'
        ).count()
        negative = TextClassification.objects.filter(
            predicted_class__icontains='negative'
        ).count()
        
        return Response({
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_percentage': (positive / total * 100) if total > 0 else 0,
            'negative_percentage': (negative / total * 100) if total > 0 else 0
        })
```

### Router Configuration

```python
# ai_app/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TextClassificationViewSet

router = DefaultRouter()
router.register(r'classifications', TextClassificationViewSet, basename='classification')

urlpatterns = [
    path('', include(router.urls)),
]
```

---

## Complete Project Examples

### Project 3: Image Classification API

```python
# ai_app/views.py
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import io
import torch
from torchvision import transforms
from .ml_models import AIModelManager

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def classify_image(request):
    """Classify uploaded image"""
    try:
        if 'image' not in request.FILES:
            return Response(
                {'error': 'Image file is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['image']
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        # Get model
        model = AIModelManager.get_image_classifier()
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Load ImageNet classes
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Format results
        results = []
        for i in range(5):
            results.append({
                'class': classes[top5_idx[i]],
                'confidence': float(top5_prob[i])
            })
        
        return Response({
            'filename': image_file.name,
            'predictions': results
        })
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

### Project 4: Text Summarization with Celery

```python
# ai_app/tasks.py (Celery tasks)
from celery import shared_task
from transformers import pipeline
from .models import TextClassification

@shared_task
def summarize_text_async(text_id, text):
    """Async text summarization"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        
        # Update or create record
        classification, created = TextClassification.objects.get_or_create(
            id=text_id,
            defaults={'text': text}
        )
        classification.summary = summary[0]['summary_text']
        classification.save()
        
        return summary[0]['summary_text']
    except Exception as e:
        return str(e)

# ai_app/views.py
from .tasks import summarize_text_async

@api_view(['POST'])
def summarize_text(request):
    """Summarize text asynchronously"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return Response(
                {'error': 'Text is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create record
        classification = TextClassification.objects.create(text=text)
        
        # Start async task
        task = summarize_text_async.delay(classification.id, text)
        
        return Response({
            'task_id': task.id,
            'status': 'processing',
            'message': 'Summarization started'
        })
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

---

## Best Practices

### 1. Model Caching

```python
# ai_app/utils.py
from functools import lru_cache
from .ml_models import AIModelManager

@lru_cache(maxsize=1)
def get_cached_model():
    """Cache model in memory"""
    return AIModelManager.get_sentiment_model()
```

### 2. Error Handling

```python
# ai_app/views.py
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
def safe_ai_prediction(request):
    """AI prediction with proper error handling"""
    try:
        # Your AI code
        pass
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return Response(
            {'error': 'Invalid input'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"AI prediction error: {e}")
        return Response(
            {'error': 'Prediction failed'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

### 3. Rate Limiting

```python
# ai_app/views.py
from django.core.cache import cache
from django.http import JsonResponse

def rate_limit_check(request, limit=10, period=60):
    """Simple rate limiting"""
    ip = request.META.get('REMOTE_ADDR')
    key = f'rate_limit_{ip}'
    count = cache.get(key, 0)
    
    if count >= limit:
        return JsonResponse(
            {'error': 'Rate limit exceeded'},
            status=429
        )
    
    cache.set(key, count + 1, period)
    return None
```

### 4. Admin Integration

```python
# ai_app/admin.py
from django.contrib import admin
from .models import TextClassification

@admin.register(TextClassification)
class TextClassificationAdmin(admin.ModelAdmin):
    list_display = ['text', 'predicted_class', 'confidence', 'created_at']
    list_filter = ['predicted_class', 'created_at']
    search_fields = ['text']
    readonly_fields = ['created_at']
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

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run server
CMD ["gunicorn", "ai_project.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Gunicorn Configuration

```python
# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
```

### Environment Variables

```bash
# .env
DEBUG=False
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=your-domain.com
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379/0
```

---

## Complete Example: AI-Powered Blog

```python
# ai_app/models.py
from django.db import models
from django.contrib.auth.models import User

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    ai_summary = models.TextField(blank=True)
    ai_sentiment = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

# ai_app/views.py
@api_view(['POST'])
def create_blog_post(request):
    """Create blog post with AI analysis"""
    title = request.data.get('title')
    content = request.data.get('content')
    
    # Create post
    post = BlogPost.objects.create(
        title=title,
        content=content,
        author=request.user
    )
    
    # AI analysis
    sentiment_model = AIModelManager.get_sentiment_model()
    sentiment = sentiment_model(content)[0]
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content, max_length=100, min_length=30)[0]
    
    # Update post
    post.ai_sentiment = sentiment['label']
    post.ai_summary = summary['summary_text']
    post.save()
    
    return Response({
        'id': post.id,
        'title': post.title,
        'ai_summary': post.ai_summary,
        'ai_sentiment': post.ai_sentiment
    })
```

---

## Key Takeaways

1. **Use Django REST Framework** for API endpoints
2. **Centralize AI models** in a manager class
3. **Use Celery** for async AI tasks
4. **Cache models** to avoid reloading
5. **Handle errors** properly
6. **Use database** to store AI predictions
7. **Admin interface** for managing AI data

---

**Master Django AI development with these complete examples!**

