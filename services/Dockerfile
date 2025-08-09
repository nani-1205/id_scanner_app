# services/Dockerfile

FROM python:3.10-slim

# --- NEW: Install essential system libraries for OpenCV and other packages ---
# libgl1-mesa-glx is the package that provides the missing libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The CMD to run the specific service is provided in the docker-compose.yml file.