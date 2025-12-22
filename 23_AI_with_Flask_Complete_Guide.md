# AI Development with Flask - Complete Guide

## 📖 Table of Contents
1. [Flask Setup for AI](#flask-setup-for-ai)
2. [AI Models Integration](#ai-models-integration)
3. [Building AI APIs](#building-ai-apis)
4. [Advanced Features](#advanced-features)
5. [Complete Project Examples](#complete-project-examples)
6. [Best Practices](#best-practices)
7. [Deployment](#deployment)

---

## Flask Setup for AI

### Project Structure

```
ai_flask_project/
├── app.py
├── requirements.txt
├── config.py
├── ml_models/
│   ├── __init__.py
│   └── models.py
├── routes/
│   ├── __init__.py
│   ├── sentiment.py
│   ├── classification.py
│   └── image.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── static/
└── templates/
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Flask and AI libraries
pip install flask flask-restful flask-cors
pip install transformers torch scikit-learn joblib
pip install flask-limiter  # Rate limiting
pip install flask-caching  # Caching
```

### requirements.txt

```txt
Flask==3.0.0
flask-restful==0.3.10
flask-cors==4.0.0
flask-limiter==3.5.0
flask-caching==2.1.0
transformers==4.35.0
torch==2.1.0
scikit-learn==1.3.2
joblib==1.3.2
Pillow==10.1.0
numpy==1.24.3
pandas==2.1.3
Werkzeug==3.0.1
```

### Basic Flask App

```python
# app.py
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# CORS
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
def home():
    return jsonify({
        'message': 'AI API is running',
        'endpoints': ['/sentiment', '/classify', '/image']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## AI Models Integration

### ML Models Manager

```python
# ml_models/models.py
import os
import joblib
import torch
from transformers import pipeline
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class AIModelManager:
    """Centralized AI model manager with lazy loading"""
    
    _models = {}
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_sentiment_model(cls):
        """Load sentiment analysis model"""
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
# routes/sentiment.py
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from ml_models.models import model_manager
import logging

logger = logging.getLogger(__name__)

sentiment_bp = Blueprint('sentiment', __name__)

@sentiment_bp.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_sentiment():
    """Analyze sentiment of text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        if not isinstance(text, str) or len(text) == 0:
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        # Get model and analyze
        model = model_manager.get_sentiment_model()
        result = model(text)[0]
        
        return jsonify({
            'text': text,
            'sentiment': result['label'].lower(),
            'confidence': result['score']
        }), 200
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@sentiment_bp.route('/analyze/batch', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_sentiment_batch():
    """Analyze sentiment of multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Texts must be a non-empty array'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Too many texts (max 100)'}), 400
        
        # Get model
        model = model_manager.get_sentiment_model()
        results = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            result = model(text)[0]
            results.append({
                'text': text,
                'sentiment': result['label'].lower(),
                'confidence': result['score']
            })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        logger.error(f"Batch sentiment analysis error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Register blueprint in app.py
# app.register_blueprint(sentiment_bp, url_prefix='/sentiment')
```

### Register Blueprints

```python
# app.py
from routes.sentiment import sentiment_bp
from routes.classification import classification_bp
from routes.image import image_bp

app.register_blueprint(sentiment_bp, url_prefix='/api/sentiment')
app.register_blueprint(classification_bp, url_prefix='/api/classification')
app.register_blueprint(image_bp, url_prefix='/api/image')
```

---

## Advanced Features

### Project 2: Text Classification with Caching

```python
# routes/classification.py
from flask import Blueprint, request, jsonify
from flask_caching import Cache
from ml_models.models import model_manager
import hashlib
import json

classification_bp = Blueprint('classification', __name__)
cache = Cache()

def get_cache_key(data):
    """Generate cache key from request data"""
    key_data = json.dumps(data, sort_keys=True)
    return f"classification:{hashlib.md5(key_data.encode()).hexdigest()}"

@classification_bp.route('/classify', methods=['POST'])
@cache.cached(timeout=3600, key_prefix=get_cache_key)
def classify_text():
    """Classify text with caching"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Get model
        model = model_manager.get_sentiment_model()
        result = model(text)[0]
        
        return jsonify({
            'text': text,
            'class': result['label'],
            'confidence': result['score']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Project 3: Image Classification

```python
# routes/image.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import torch
from torchvision import transforms
from ml_models.models import model_manager
import os

image_bp = Blueprint('image', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@image_bp.route('/classify', methods=['POST'])
def classify_image():
    """Classify uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
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
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Load ImageNet classes
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Format results
        predictions = []
        for i in range(5):
            predictions.append({
                'class': classes[top5_idx[i]],
                'confidence': float(top5_prob[i])
            })
        
        return jsonify({
            'filename': file.filename,
            'predictions': predictions
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Complete Project Examples

### Project 4: AI Service with Database

```python
# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'prediction_type': self.prediction_type,
            'result': self.result,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }

# app.py
from models import db, Prediction

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# routes/predictions.py
from models import db, Prediction
from ml_models.models import model_manager

@prediction_bp.route('/predict', methods=['POST'])
def predict_and_save():
    """Predict and save to database"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Get model and predict
        model = model_manager.get_sentiment_model()
        result = model(text)[0]
        
        # Save to database
        prediction = Prediction(
            text=text,
            prediction_type='sentiment',
            result=result['label'],
            confidence=result['score']
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify(prediction.to_dict()), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        predictions = Prediction.query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return jsonify({
            'predictions': [p.to_dict() for p in predictions.items],
            'total': predictions.total,
            'pages': predictions.pages,
            'current_page': page
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Project 5: Text Summarization

```python
# routes/summarization.py
from flask import Blueprint, request, jsonify
from ml_models.models import model_manager

summarization_bp = Blueprint('summarization', __name__)

@summarization_bp.route('/summarize', methods=['POST'])
def summarize_text():
    """Summarize text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) < min_length:
            return jsonify({'error': f'Text too short (min {min_length} characters)'}), 400
        
        # Get model
        summarizer = model_manager.get_summarizer()
        
        # Summarize
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return jsonify({
            'original_text': text,
            'summary': summary[0]['summary_text'],
            'original_length': len(text),
            'summary_length': len(summary[0]['summary_text']),
            'compression_ratio': len(summary[0]['summary_text']) / len(text)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Best Practices

### 1. Error Handling

```python
# app.py
from flask import jsonify

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413
```

### 2. Request Validation

```python
# utils/validators.py
def validate_text_request(data):
    """Validate text request"""
    if not data:
        return False, 'No data provided'
    
    if 'text' not in data:
        return False, 'Text field is required'
    
    text = data['text']
    if not isinstance(text, str):
        return False, 'Text must be a string'
    
    if len(text) == 0:
        return False, 'Text cannot be empty'
    
    if len(text) > 5000:
        return False, 'Text too long (max 5000 characters)'
    
    return True, None

# Usage in routes
@sentiment_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    is_valid, error = validate_text_request(data)
    
    if not is_valid:
        return jsonify({'error': error}), 400
    
    # Continue with processing
```

### 3. Logging

```python
# app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/ai_api.log',
        maxBytes=10240000,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('AI API startup')
```

### 4. Configuration Management

```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# app.py
from config import config

app.config.from_object(config[os.getenv('FLASK_ENV', 'default')])
```

---

## Deployment

### Gunicorn Configuration

```python
# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
```

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
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./ml_models:/app/ml_models
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
```

---

## Key Takeaways

1. **Flask is simple and flexible** - Easy to learn and customize
2. **Use blueprints** - Organize routes
3. **Add rate limiting** - Protect your API
4. **Use caching** - Improve performance
5. **Validate requests** - Ensure data quality
6. **Handle errors** - Proper error responses
7. **Use database** - Store predictions

---

**Master Flask AI development with these complete examples!**

