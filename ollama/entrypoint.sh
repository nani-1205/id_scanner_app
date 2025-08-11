#!/bin/sh

# Start the Ollama server in the background so we can run commands against it.
/bin/ollama serve &
pid=$!

# Wait for the server to be fully ready.
echo "Waiting for Ollama server to be fully ready..."
while ! ollama list > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo "Ollama server is up and ready to accept commands."

# Define the model name we want to create and the source file.
MODEL_NAME="phi3-vision"
GGUF_FILE="Phi-3-vision-128k-instruct-Q4_K_M.gguf"
GGUF_URL="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-gguf/resolve/main/${GGUF_FILE}"

# Check if the model already exists. If not, build it.
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "'$MODEL_NAME' model not found. Building from Hugging Face..."

    echo "Downloading model file from ${GGUF_URL}..."
    curl -L "$GGUF_URL" -o "$GGUF_FILE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download model file."
        exit 1
    fi
    echo "Download complete."

    echo "Creating model '$MODEL_NAME' using Modelfile..."
    ollama create "$MODEL_NAME" -f /Modelfile
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create model with ollama create."
        exit 1
    fi
    echo "Model creation successful."

    # Clean up the downloaded file after creating the model to save space.
    rm "$GGUF_FILE"
else
    echo "'$MODEL_NAME' model already exists."
fi

echo "The required vision model is available."
echo "Ollama is running in the background. Container will remain active."

# Wait for the server process to exit to keep the container alive.
wait $pid