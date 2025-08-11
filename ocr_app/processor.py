import cv2
import numpy as np
import requests
import base64
import json
import pytesseract
from PIL import Image
import io
from celery import shared_task
from database import save_processed_document

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# We now use a pure language model for structuring text
LANGUAGE_MODEL = "llama3"

def extract_raw_text_with_tesseract(image_bytes_list):
    """
    Step 1: Use Tesseract (a specialized OCR engine) to extract all text from images.
    This is much more accurate for pure text recognition than a multimodal model.
    """
    full_text = ""
    for image_bytes in image_bytes_list:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Pre-processing can be added here (e.g., grayscale, thresholding) if needed
            text = pytesseract.image_to_string(image, lang='eng')
            full_text += text + "\n\n"
        except Exception as e:
            print(f"Error during Tesseract OCR: {e}")
    return full_text

def structure_text_with_llama3(raw_text, doc_type):
    """
    Step 2: Use a powerful language model (Llama 3) to structure the raw text.
    The model's only job is to understand and organize, not to see.
    """
    if doc_type != "Driving License":
        return {"error": "This document type is not yet supported by the structuring engine."}
    
    prompt = f"""
    You are an expert data extraction assistant. Below is raw text extracted from a Philippine Driver's License.
    Your task is to parse this text and populate the following JSON structure.
    Fill every field with the corresponding data from the text.
    If data for a field is not present in the text, leave the value as an empty string "".
    Do NOT add any commentary, notes, or markdown. Respond ONLY with the JSON object.

    JSON Structure to populate:
    {{
      "lastName": "",
      "firstName": "",
      "middleName": "",
      "nationality": "",
      "sex": "",
      "dateOfBirth": "YYYY/MM/DD",
      "weightKg": "",
      "heightM": "",
      "address": "",
      "licenseNo": "",
      "expirationDate": "YYYY/MM/DD",
      "agencyCode": "",
      "bloodType": "",
      "eyesColor": "",
      "dlCodes": "",
      "conditions": "",
      "serialNumber": ""
    }}

    --- Raw Text from OCR ---
    {raw_text}
    """
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": LANGUAGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json" # Crucial for getting a clean JSON response
            },
            timeout=180
        )
        response.raise_for_status()
        response_data = response.json().get('response', '{}')
        return json.loads(response_data)
    except Exception as e:
        print(f"Error structuring text with Llama3: {e}")
        return {"error": f"The language model failed to structure the text. Error: {e}"}

@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """Celery task using the Tesseract + Llama 3 pipeline."""
    try:
        image_bytes_list = list(file_contents.values())
        
        # --- Step 1: High-Accuracy OCR with Tesseract ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR with Tesseract...'})
        raw_text = extract_raw_text_with_tesseract(image_bytes_list)
        if not raw_text.strip():
            raise Exception("Tesseract failed to extract any text from the document.")

        # --- Step 2: Intelligent Structuring with Llama 3 ---
        self.update_state(state='PROGRESS', meta={'status': f'Structuring text with {LANGUAGE_MODEL} model...'})
        final_data = structure_text_with_llama3(raw_text, doc_type)
        if "error" in final_data:
            raise Exception(final_data["error"])

        # --- Final Steps ---
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(image_bytes_list)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(doc_type, json_data, image_bytes_list, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        self.update_state(state='FAILURE', meta={'status': str(e)})
        raise e

def detect_and_crop_face(image_bytes_list):
    """Finds a face from any of the provided images."""
    for img_bytes in image_bytes_list:
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_crop = img[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.jpg', face_crop)
                return buffer.tobytes()
        except Exception as e:
            print(f"Error during face detection: {e}")
            continue
    return None