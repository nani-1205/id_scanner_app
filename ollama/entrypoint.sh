#!/bin/sh
# Start the Ollama server in the background
/bin/ollama serve &
pid=$!

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s --fail -o /dev/null http://localhost:11434; do
    sleep 1
    echo -n "."
done
echo "Ollama server is up and running."

# --- PULL REQUIRED MODELS ---
# List of models our application needs
REQUIRED_MODELS="llava bakllava"

for model in $REQUIRED_MODELS; do
    if ! ollama list | grep -q "$model"; then
        echo "'$model' model not found. Pulling..."
        ollama pull "$model"
    else
        echo "'$model' model already exists."
    fi
done

echo "All required models are available."

# Wait for the server process to exit to keep the container alive
wait $pid