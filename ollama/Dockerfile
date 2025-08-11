FROM ollama/ollama

# Copy the custom entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Ensure it's executable
RUN chmod +x /entrypoint.sh

# Set it as the entrypoint for this container
ENTRYPOINT ["/entrypoint.sh"]