# We explicitly specify the :latest tag to bypass any old caches.
FROM ollama/ollama:latest

# We only need curl for the health check in the entrypoint script.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container (optional, but good practice)
WORKDIR /

# <<< THE FIX IS HERE >>>
# Removed unnecessary lines:
# - COPY requirements.txt .
# - RUN pip install --no-cache-dir -r requirements.txt
# - Anything else related to Python packages as this is NOT a Python app.

# Copy ONLY the entrypoint script.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]