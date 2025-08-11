FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# MODIFIED THIS LINE: Added the --timeout flag
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "app:app"]