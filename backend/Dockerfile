FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache HF models by copying download script first and running it
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run application (seeding default users first, then starting uvicorn)
CMD ["sh", "-c", "python seed_users.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
