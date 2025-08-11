#!/bin/sh
# Start the Ollama server in the background
/bin/ollama serve &
pid=$!

# Wait for the server to be ready by using its own CLI tool.
echo "Waiting for Ollama server to be fully ready..."
while ! ollama list > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo "Ollama server is up and ready to accept commands."

# --- PULL THE REQUIRED VISION MODEL ---
REQUIRED_MODEL="llava-phi3"

if ! ollama list | grep -q "$REQUIRED_MODEL"; then
    echo "'$REQUIRED_MODEL' model not found. Pulling directly with Ollama..."
    ollama pull "$REQUIRED_MODEL"
    if [ $? -ne 0 ]; then
        echo "ERROR: 'ollama pull' command failed. Please check the model name and network connection."
        exit 1
    fi
else
    echo "'$REQUIRED_MODEL' model already exists."
fi

echo "The required vision model is now available."
echo "Ollama is running in the background. Container will remain active."

# Wait for the server process to exit to keep the container alive.
wait $pid