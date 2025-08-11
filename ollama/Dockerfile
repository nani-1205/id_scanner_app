# We explicitly specify the :latest tag to bypass any old caches.
FROM ollama/ollama:latest

# We need curl for downloading the model file in the entrypoint script.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy BOTH the Modelfile and the entrypoint script into the container.
COPY Modelfile /Modelfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint for the container.
ENTRYPOINT ["/entrypoint.sh"]