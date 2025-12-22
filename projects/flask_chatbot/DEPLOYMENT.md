# Flask Chatbot - Deployment Guide

## 🚀 Deployment Options

### 1. Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - USE_OLLAMA=false
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    restart: unless-stopped
```

#### Build and Run
```bash
docker build -t flask-chatbot .
docker run -p 5000:5000 flask-chatbot
```

---

### 2. Heroku Deployment

#### Procfile
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

#### runtime.txt
```
python-3.9.18
```

#### Deploy Steps
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-flask-chatbot

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set FLASK_ENV=production
heroku config:set HUGGINGFACE_API_KEY=your-key

# Deploy
git init
git add .
git commit -m "Initial commit"
git push heroku main

# Open app
heroku open
```

---

### 3. AWS EC2 with Gunicorn + Nginx

#### Gunicorn Configuration
```python
# gunicorn_config.py
bind = "127.0.0.1:5000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
```

#### Systemd Service
```ini
# /etc/systemd/system/flask-chatbot.service
[Unit]
Description=Flask Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/flask_chatbot
Environment="PATH=/home/ubuntu/flask_chatbot/venv/bin"
ExecStart=/home/ubuntu/flask_chatbot/venv/bin/gunicorn \
    --config gunicorn_config.py \
    app:app

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/flask-chatbot
upstream flask {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://flask;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

### 4. Railway Deployment

#### Deploy Steps
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Set environment variables
railway variables set FLASK_ENV=production
railway variables set SECRET_KEY=your-secret-key

# Deploy
railway up
```

---

## 🔧 Production Configuration

### app.py (Production)
```python
import os

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
```

### Production Settings
```python
# config.py
import os

class ProductionConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-in-production'
    DEBUG = False
    TESTING = False

# app.py
app.config.from_object(ProductionConfig)
```

---

## ✅ Deployment Checklist

- [ ] Set FLASK_ENV=production
- [ ] Set SECRET_KEY
- [ ] Configure CORS for production
- [ ] Set up SSL/HTTPS
- [ ] Configure logging
- [ ] Test all endpoints
- [ ] Load testing

---

**Flask is simple and easy to deploy for AI applications!**

