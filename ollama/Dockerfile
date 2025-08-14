# Start from the absolute latest official Ollama image
FROM ollama/ollama:latest

# Install curl, which is needed by our entrypoint script for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy our custom entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh
# Make the script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint to our custom script
ENTRYPOINT ["/entrypoint.sh"]