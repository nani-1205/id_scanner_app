# services/Dockerfile

FROM python:3.10-slim

# Install essential system libraries required by OpenCV, TensorFlow, and Pillow
# This prevents common runtime errors like "libGL.so.1: cannot open shared object file"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

# Install Python packages using pip, with an increased timeout for large libraries
# like TensorFlow.
RUN pip --default-timeout=300 install --no-cache-dir -r requirements.txt

# The CMD to run the specific service (web-app or worker) is provided in the docker-compose.yml file.