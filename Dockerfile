# PaddyTwin: the twin API and the dashboard (CPU build; the models run on CPU at field scale).
#   docker compose up --build      ->  dashboard http://localhost:8501, API http://localhost:8008/docs
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p outputs
EXPOSE 8008 8501
CMD ["python", "digital_twin_api.py"]
