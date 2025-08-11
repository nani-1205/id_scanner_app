# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Run the app using gunicorn for a production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]