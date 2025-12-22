# FastAPI Chatbot - Deployment Guide

## 🚀 Deployment Options

### 1. Docker Deployment (Recommended)

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
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - USE_OLLAMA=false
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    volumes:
      - ./static:/app/static
      - ./templates:/app/templates
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

#### Build and Run
```bash
# Build image
docker build -t fastapi-chatbot .

# Run container
docker run -p 8000:8000 fastapi-chatbot

# Or use docker-compose
docker-compose up -d
```

---

### 2. Heroku Deployment

#### Procfile
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
```

#### runtime.txt
```
python-3.9.18
```

#### Deploy Steps
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-chatbot-name

# Set environment variables
heroku config:set HUGGINGFACE_API_KEY=your-key
heroku config:set USE_OLLAMA=false

# Deploy
git init
git add .
git commit -m "Initial commit"
git push heroku main

# Open app
heroku open
```

---

### 3. AWS EC2 Deployment

#### Setup Script
```bash
#!/bin/bash
# setup.sh

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install Nginx
sudo apt-get install -y nginx

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Gunicorn
pip install gunicorn

# Create systemd service
sudo nano /etc/systemd/system/chatbot.service
```

#### Systemd Service File
```ini
[Unit]
Description=FastAPI Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fastapi_chatbot
Environment="PATH=/home/ubuntu/fastapi_chatbot/venv/bin"
ExecStart=/home/ubuntu/fastapi_chatbot/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/chatbot
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /home/ubuntu/fastapi_chatbot/static;
    }
}
```

#### Enable and Start
```bash
# Enable service
sudo systemctl enable chatbot
sudo systemctl start chatbot

# Enable Nginx
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

### 4. Railway Deployment

#### railway.json
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### Deploy Steps
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

---

### 5. DigitalOcean App Platform

#### app.yaml
```yaml
name: fastapi-chatbot
services:
- name: api
  source_dir: /
  github:
    repo: your-username/fastapi-chatbot
    branch: main
  run_command: uvicorn main:app --host 0.0.0.0 --port $PORT
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: HUGGINGFACE_API_KEY
    value: your-key
    scope: RUN_TIME
```

---

## 🔧 Production Configuration

### main.py (Production)
```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
```

### Environment Variables
```bash
# .env
USE_OLLAMA=false
HUGGINGFACE_API_KEY=your-key
DEBUG=false
LOG_LEVEL=info
```

---

## 📊 Monitoring

### Health Check Endpoint
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "FastAPI Chatbot",
        "version": "1.0.0"
    }
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
```

---

## ✅ Deployment Checklist

- [ ] Set environment variables
- [ ] Configure CORS for production domain
- [ ] Set up SSL/HTTPS
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Configure backup strategy
- [ ] Test health endpoints
- [ ] Load testing
- [ ] Security review

---

**FastAPI is excellent for AI deployment due to its speed and async capabilities!**

