FROM python:3.11-slim AS build

RUN apt-get update -qq && apt-get install -y -qq \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

RUN apt-get update -qq && apt-get install -y -qq \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

COPY . .

ENV SERVE_HOST=0.0.0.0
ENV SERVE_PORT=8000
ENV SERVE_CHECKPOINT_PATH=/app/checkpoints/best_model.safetensors

EXPOSE 8000

CMD ["python", "-m", "serve.run"]
