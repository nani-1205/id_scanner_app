#!/bin/sh

# Start the Ollama server in the background
/bin/ollama serve &
pid=$!

# Wait for the server to be ready by polling its status endpoint
echo "Waiting for Ollama server to start..."
while ! curl -s --fail -o /dev/null http://localhost:11434; do
    sleep 1
    echo -n "."
done
echo "Ollama server is up and running."

# Check if the 'llava' model already exists. If not, pull it.
if ! ollama list | grep -q "llava"; then
    echo "'llava' model not found. Pulling the model..."
    ollama pull llava
    echo "Model 'llava' pulled successfully."
else
    echo "'llava' model already exists."
fi

# Wait for the server process to exit to keep the container alive
wait $pid