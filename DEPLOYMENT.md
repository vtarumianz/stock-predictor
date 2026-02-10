# Deployment Guide 🚀

This guide covers deploying your Stock Predictor app to various cloud platforms.

## Table of Contents
1. [Streamlit Cloud](#streamlit-cloud) - **Recommended (Free!)**
2. [Heroku](#heroku)
3. [Railway](#railway)
4. [Render](#render)
5. [Docker](#docker)

---

## Streamlit Cloud

### ✨ Why Streamlit Cloud?
- **Free** - Hobby tier is completely free
- **Easy** - Deploy with 1 click
- **Fast** - Auto-deploys on every git push
- **Built for Streamlit** - Perfect integration

### Step-by-Step

#### 1. Push Code to GitHub

```bash
# Create repository on GitHub (if you haven't already)
# Go to https://github.com/new
# Name: stock-predictor
# Make it Public
# Don't initialize with README

# Initialize git and push
git init
git add .
git commit -m "Initial commit: Stock Predictor app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/stock-predictor.git
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in with GitHub" - Login with your GitHub account
3. Click "New app"
4. Fill in the form:
   - **Repository**: YOUR-USERNAME/stock-predictor
   - **Branch**: main
   - **Main file path**: app.py
5. Click "Deploy"
6. Wait 2-3 minutes for deployment
7. Your app is live! 🎉

**Your app URL**: `https://YOUR-USERNAME-stock-predictor.streamlit.app`

### Auto-Deployment

After initial deployment, your app automatically updates whenever you:
```bash
git add .
git commit -m "Your changes"
git push
```

---

## Heroku

### Prerequisites
- Heroku account (free at [heroku.com](https://www.heroku.com))
- Heroku CLI ([download](https://devcenter.heroku.com/articles/heroku-cli))

### Step-by-Step

#### 1. Create Procfile (Already included)

The project already has a Procfile:
```
web: streamlit run --server.port=$PORT app.py
```

#### 2. Create .slugignore (if needed)

```bash
echo ".git
.gitignore
README.md
docs/" > .slugignore
```

#### 3. Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-stock-predictor

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

#### 4. View Logs

```bash
heroku logs --tail
```

### Cost Notes
- Free tier is limited and may go to sleep
- Consider paid tier for continuous uptime ($7+/month)

---

## Railway

### Prerequisites
- Railway account (free at [railway.app](https://railway.app))

### Step-by-Step

#### 1. Connect GitHub

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "Create New Project"
4. Select "Deploy from GitHub repo"
5. Select your `stock-predictor` repository
6. Click "Deploy"

#### 2. Configure Environment

Railway auto-detects Python and builds from `requirements.txt`

#### 3. Custom Domain (Optional)

1. Go to Settings
2. Add custom domain
3. Configure DNS

### Cost Notes
- Free tier includes $5/month free credit
- Great for hobby projects

---

## Render

### Prerequisites
- Render account (free at [render.com](https://render.com))

### Step-by-Step

#### 1. Connect GitHub

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +"
4. Select "Web Service"
5. Select your repository
6. Fill in details:
   - Name: `stock-predictor`
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py`

#### 2. Set Environment Variables

Add to Render (Environment section):
```
STREAMLIT_SERVER_PORT=10000
```

#### 3. Deploy

- Click "Create Web Service"
- Render auto-deploys on every git push

### Cost Notes
- Free tier available
- Sleeps after 15 minutes of inactivity
- Paid tier for production ($7-12/month)

---

## Docker

### Build and Run Locally

#### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

#### 2. Build Image

```bash
docker build -t stock-predictor .
```

#### 3. Run Container

```bash
docker run -p 8501:8501 stock-predictor
```

Open [http://localhost:8501](http://localhost:8501)

### Deploy Docker to Cloud

#### Option A: GitHub Actions (CI/CD)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Docker Hub

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: YOUR-USERNAME/stock-predictor:latest
```

#### Option B: AWS, Google Cloud, Azure

See cloud provider documentation for container deployment.

---

## Comparison Table

| Platform | Cost | Ease | Performance | Auto-Deploy |
|----------|------|------|-------------|------------|
| Streamlit Cloud | Free | ⭐⭐⭐⭐⭐ | Good | ✅ |
| Heroku | $7+/month | ⭐⭐⭐⭐ | Good | ✅ |
| Railway | Free* | ⭐⭐⭐⭐ | Good | ✅ |
| Render | Free* | ⭐⭐⭐⭐ | Good | ✅ |
| Docker | Variable | ⭐⭐⭐ | Great | ❌ |

*Free tier with limitations

---

## Troubleshooting

### "ModuleNotFoundError"
- Check `requirements.txt` is in root directory
- Ensure all dependencies are listed

### App is slow on free tier
- Consider upgrading to paid tier
- Free tier has limited resources

### Deployment fails
- Check logs for error messages
- Verify Python version (3.8+)
- Ensure all files are committed to git

### Custom domain issues
- Allow DNS propagation (up to 24 hours)
- Check DNS settings with your domain provider

---

## Git Workflow

### Initial Setup
```bash
git init
git add .
git commit -m "Initial commit: Stock Predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/stock-predictor.git
git push -u origin main
```

### Regular Updates
```bash
git add .
git commit -m "Description of changes"
git push
# App auto-deploys!
```

---

## Recommended: Streamlit Cloud

For easiest setup, fastest deployment, and best free tier, I recommend **Streamlit Cloud**:

1. Push to GitHub
2. Go to share.streamlit.io
3. Click "New app"
4. Select repository → Deploy
5. Done! 🎉

Your app will be live in 2-3 minutes and auto-update when you push to GitHub.

---

## Next Steps

- Set up custom domain (optional)
- Enable authentication (optional)
- Monitor performance and logs
- Add more features and iterate!

Happy deploying! 🚀
