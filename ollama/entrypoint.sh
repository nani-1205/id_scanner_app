#!/bin/sh
# Start the Ollama server in the background
/bin/ollama serve &
pid=$!

# Wait for the server to be ready by using its own CLI tool.
# This is the most reliable way to check if the API is fully operational.
echo "Waiting for Ollama server to be fully ready..."
while ! ollama list > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo "Ollama server is up and ready to accept commands."

# --- PULL THE REQUIRED VISION MODEL ---
REQUIRED_MODEL="phi3:vision"

if ! ollama list | grep -q "$REQUIRED_MODEL"; then
    echo "'$REQUIRED_MODEL' model not found. Pulling..."
    ollama pull "$REQUIRED_MODEL"
else
    echo "'$REQUIRED_MODEL' model already exists."
fi

echo "Vision model is available."

# Wait for the server process to exit to keep the container alive
wait $pid