# Django Chatbot - Deployment Guide

## 🚀 Deployment Options

### 1. Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "chatbot_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=chatbot
      - POSTGRES_USER=chatbot_user
      - POSTGRES_PASSWORD=chatbot_pass
    restart: unless-stopped

  web:
    build: .
    command: gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
    ports:
      - "8000:8000"
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://chatbot_user:chatbot_pass@db:5432/chatbot
      - USE_OLLAMA=false
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    restart: unless-stopped

volumes:
  postgres_data:
  static_volume:
```

#### Build and Run
```bash
docker-compose up -d
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py createsuperuser
```

---

### 2. Heroku Deployment

#### Procfile
```
web: gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:$PORT
release: python manage.py migrate
```

#### runtime.txt
```
python-3.9.18
```

#### Deploy Steps
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-django-chatbot

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
heroku config:set HUGGINGFACE_API_KEY=your-key

# Deploy
git init
git add .
git commit -m "Initial commit"
git push heroku main

# Run migrations
heroku run python manage.py migrate
heroku run python manage.py createsuperuser

# Open app
heroku open
```

---

### 3. AWS EC2 with Gunicorn + Nginx

#### Gunicorn Configuration
```python
# gunicorn_config.py
bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
```

#### Systemd Service
```ini
# /etc/systemd/system/django-chatbot.service
[Unit]
Description=Django Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/django_chatbot
Environment="PATH=/home/ubuntu/django_chatbot/venv/bin"
ExecStart=/home/ubuntu/django_chatbot/venv/bin/gunicorn \
    --config gunicorn_config.py \
    chatbot_project.wsgi:application

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/django-chatbot
upstream django {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location /static/ {
        alias /home/ubuntu/django_chatbot/staticfiles/;
    }

    location /media/ {
        alias /home/ubuntu/django_chatbot/media/;
    }

    location / {
        proxy_pass http://django;
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

# Add PostgreSQL
railway add postgresql

# Set environment variables
railway variables set SECRET_KEY=your-secret-key
railway variables set DEBUG=False

# Deploy
railway up
```

---

## 🔧 Production Settings

### settings.py (Production)
```python
import os
import dj_database_url

DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com']

# Database
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL')
    )
}

# Static files
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'

# Security
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

---

## ✅ Deployment Checklist

- [ ] Set DEBUG=False
- [ ] Configure ALLOWED_HOSTS
- [ ] Set up database (PostgreSQL recommended)
- [ ] Run migrations
- [ ] Collect static files
- [ ] Set up SSL/HTTPS
- [ ] Configure security settings
- [ ] Set up logging
- [ ] Configure admin user
- [ ] Test all endpoints

---

**Django is great for full-featured AI applications with admin panel!**

