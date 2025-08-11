FROM python:3.10-slim

# Install system dependencies, now including the missing libglib2.0-0
# I have also alphabetized the list for better maintenance.
RUN apt-get update && apt-get install -y \
    build-essential \
    libfontconfig1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libice6 \
    libsm6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# This pip install will take significantly longer due to PaddlePaddle's size.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]