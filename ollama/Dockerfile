FROM python:3.10-slim

# Install system dependencies for PaddleOCR and OpenCV.
# Removed 'libgl1-mesa-glx' (obsolete), added 'libgl1' (general GL lib),
# and removed 'tesseract-ocr' (no longer needed for PaddleOCR pipeline).
RUN apt-get update && apt-get install -y \
    build-essential \
    libfontconfig1 \
    libgl1 \
    libglib2.0-0 \
    libice6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# This pip install will take significantly longer due to PaddlePaddle's size.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--graceful-timeout", "180", "app:app"]