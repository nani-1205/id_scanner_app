FROM python:3.10-slim

# Install system dependencies, including font libraries for PaddleOCR.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# This pip install will take significantly longer due to PaddlePaddle's size.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]