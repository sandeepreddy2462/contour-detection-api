# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching pip layer)
COPY requirements.txt .

# Install Python dependencies (no cache for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (metadata only — Cloud Run overrides this)
EXPOSE 8000

# Start the Uvicorn server — dynamically use Cloud Run's $PORT
# FIX: Changed to the shell form (without []) to allow $PORT substitution.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
