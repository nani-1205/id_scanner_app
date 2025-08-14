# Start from the official Python 3.10 slim base image.
# 'slim' provides a good balance between size and available tools.
FROM python:3.10-slim

# Install system-level dependencies required by the Python libraries.
# This is a crucial step to prevent runtime errors.
RUN apt-get update && apt-get install -y \
    # build-essential: Provides compilers (gcc, g++, make) needed for some pip packages.
    build-essential \
    # libfontconfig1, libice6, libsm6, libxrender1: Font and X11 libraries used by PaddleOCR and OpenCV.
    libfontconfig1 \
    libice6 \
    libsm6 \
    libxrender1 \
    # libgl1: Core OpenGL library, a dependency for headless OpenCV operations.
    libgl1 \
    # libglib2.0-0: Provides libgthread, a critical dependency for OpenCV that was causing startup crashes.
    libglib2.0-0 \
    # Clean up apt cache to keep the final image size smaller.
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for all subsequent commands.
WORKDIR /app

# Copy only the requirements file first.
# This leverages Docker's layer caching. If requirements.txt doesn't change,
# Docker won't re-run the time-consuming 'pip install' step on subsequent builds.
COPY requirements.txt .

# Install all Python dependencies from the requirements file.
# --no-cache-dir: Reduces the final image size by not storing the pip download cache.
# This step will take a significant amount of time due to the size of paddlepaddle.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code into the container.
COPY . .

# Define the command to run when the container starts.
# We use Gunicorn, a production-grade WSGI server, to run the Flask app.
# --bind 0.0.0.0:5000: Binds to all network interfaces inside the container on port 5000.
# --timeout 180: Allows workers up to 180 seconds to complete a single request.
# --graceful-timeout 180: Gives workers up to 180 seconds to boot up before Gunicorn kills them.
#   This is critical because PaddleOCR initialization is very slow on the first start.
# app:app: Tells Gunicorn to run the 'app' object from the 'app.py' module.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--graceful-timeout", "180", "app:app"]