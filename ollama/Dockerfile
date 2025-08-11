# We explicitly specify the :latest tag to bypass any old caches.
FROM ollama/ollama:latest

# We still need curl for the health check in the entrypoint script.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script that contains the model pull logic.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint for the container.
ENTRYPOINT ["/entrypoint.sh"]