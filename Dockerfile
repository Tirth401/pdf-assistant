FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST (saves ~1.8GB vs GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (torch already satisfied, won't re-install GPU version)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Railway sets PORT env var automatically
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
