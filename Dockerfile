FROM python:3.11-slim

# HF Spaces metadata
LABEL maintainer="hackathon-participant"
LABEL space-sdk="gradio"
LABEL tags="openenv"

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port (primary) and Gradio port
EXPOSE 7860
EXPOSE 7861

# Environment defaults (override at runtime)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_URL="http://localhost:7860"

CMD ["python", "main.py"]
