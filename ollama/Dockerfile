FROM ollama/ollama

# ADD THIS LINE: Update package list and install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the custom entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Ensure it's executable
RUN chmod +x /entrypoint.sh

# Set it as the entrypoint for this container
ENTRYPOINT ["/entrypoint.sh"]