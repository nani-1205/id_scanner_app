FROM python:3.10-slim

# Install system libraries for OpenCV, TensorFlow, Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

# Install Python packages with an increased timeout
RUN pip --default-timeout=300 install --no-cache-dir -r requirements.txt