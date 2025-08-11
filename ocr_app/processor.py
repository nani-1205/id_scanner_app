import cv2
import numpy as np
import requests
import base64
import json

OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0: return None

        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_crop = img[y:y+h, x:x+w]
        
        _, buffer = cv2.imencode('.jpg', face_crop)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

def extract_data_with_ollama(image_bytes_list, doc_type):
    base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]

    prompt = f"""
    You are an expert OCR system for identity documents.
    Analyze the provided image(s) of a "{doc_type}".
    Extract key information and return it as a clean, minified JSON object.
    Do NOT include any explanatory text, markdown formatting (like ```json), or any other text outside the JSON object.
    The JSON should contain keys like "firstName", "lastName", "documentNumber", "dateOfBirth", "expiryDate", etc.
    If a field is not present, omit it. For a driver's license, combine info from both front and back images.
    """

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llava",
                "prompt": prompt,
                "images": base64_images,
                "stream": False,
                "format": "json" # This tells Ollama to strictly output JSON
            },
            timeout=120
        )
        response.raise_for_status()
        
        response_data = response.json()
        extracted_json_str = response_data.get('response', '{}')

        return json.loads(extracted_json_str)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": "Failed to communicate with the AI model."}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama: {e}")
        print(f"Raw response was: {extracted_json_str}")
        return {"error": "AI model returned invalid JSON.", "raw_response": extracted_json_str}

def process_documents(files, doc_type):
    if not files or all(f.filename == '' for f in files):
        return None, "No files selected."

    image_bytes_list = [f.read() for f in files]

    face_image_bytes = None
    for img_bytes in image_bytes_list:
        face_image_bytes = detect_and_crop_face(img_bytes)
        if face_image_bytes:
            break
    
    extracted_data = extract_data_with_ollama(image_bytes_list, doc_type)

    if "error" in extracted_data:
        return None, extracted_data.get("raw_response", "An error occurred during AI processing.")

    return extracted_data, face_image_bytes, image_bytes_list