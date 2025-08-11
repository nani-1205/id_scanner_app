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

# --- PULL THE REQUIRED LANGUAGE MODEL ---
REQUIRED_MODEL="llama3"

if ! ollama list | grep -q "$REQUIRED_MODEL"; then
    echo "'$REQUIRED_MODEL' model not found. Pulling..."
    ollama pull "$REQUIRED_MODEL"
else
    echo "'$REQUIRED_MODEL' model already exists."
fi

echo "Language model is available."

# Wait for the server process to exit to keep the container alive
wait $pid