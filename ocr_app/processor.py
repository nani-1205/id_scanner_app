import cv2
import numpy as np
import requests
import base64
import json
import time

OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_bytes):
    """Detects the most prominent face in an image and returns the cropped face."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Assume the largest detected face is the correct one
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop = img[y:y+h, x:x+w]
        
        _, buffer = cv2.imencode('.jpg', face_crop)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

def extract_data_with_ollama(image_bytes_list, doc_type):
    """Uses Ollama's LLaVA model to extract structured data from images."""
    base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]

    prompt = f"""
    You are an expert OCR system for identity documents.
    Analyze the provided image(s) of a "{doc_type}".
    Extract the key information and return it as a clean, minified JSON object.
    Do NOT include any explanatory text or markdown formatting (like ```json).
    The JSON should contain keys like "firstName", "lastName", "documentNumber", "dateOfBirth", "expiryDate", etc.
    If a field is not present, omit it from the JSON.
    For a driver's license, if there's a front and back, combine the information.
    """

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llava",
                "prompt": prompt,
                "images": base64_images,
                "stream": False,  # Set to False to get the full response at once
                "format": "json"  # Ask Ollama to format the output as JSON
            },
            timeout=120  # LLaVA can be slow, use a long timeout
        )
        response.raise_for_status()
        
        # The response text should be a JSON string
        response_data = response.json()
        # The actual content is in the 'response' key, which is a JSON string itself
        extracted_json_str = response_data.get('response', '{}')

        # Parse the JSON string from the response
        return json.loads(extracted_json_str)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": "Failed to communicate with the AI model."}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama: {e}")
        print(f"Raw response was: {extracted_json_str}")
        return {"error": "AI model returned invalid JSON.", "raw_response": extracted_json_str}


def process_documents(files, doc_type):
    """Main processing pipeline."""
    if not files or all(f.filename == '' for f in files):
        return None, "No files selected."

    image_bytes_list = [f.read() for f in files]

    # Find a face from any of the uploaded images
    face_image_bytes = None
    for img_bytes in image_bytes_list:
        face_image_bytes = detect_and_crop_face(img_bytes)
        if face_image_bytes:
            break
    
    # Extract data using Ollama
    extracted_data = extract_data_with_ollama(image_bytes_list, doc_type)

    if "error" in extracted_data:
        return None, extracted_data.get("raw_response", "An error occurred during AI processing.")

    return extracted_data, face_image_bytes, image_bytes_list