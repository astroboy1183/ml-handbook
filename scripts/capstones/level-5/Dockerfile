# Level 5 capstone: the churn model service.
# Build after training (python train.py), from this directory:
#   docker build -t churn-api:1.0.0 .
#   docker run --rm -p 8000:8000 churn-api:1.0.0
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# Dependencies first, so this layer stays cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service code and the trained model artifact
COPY app.py .
COPY model/model.joblib model/metadata.json ./model/

# Run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
