import cv2
import numpy as np
import requests
import base64
import json
from celery import shared_task
from database import save_processed_document

OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def generate_prompt(doc_type):
    """Generates a highly specific prompt based on the document type."""
    
    # --- PROMPT FOR PHILIPPINE DRIVER'S LICENSE ---
    if doc_type == "Driving License":
        return """
        You are a highly accurate OCR data extraction expert specializing in Philippine Driver's Licenses.
        Your task is to analyze the provided front and back images of the license.
        Extract the information and populate the following JSON structure precisely.
        Do NOT add any fields that are not in this structure.
        Do NOT add commentary, notes, or markdown.
        If you cannot find information for a field, leave its value as an empty string "".

        {
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
        }
        """

    # --- GENERIC PROMPT FOR OTHER DOCUMENT TYPES ---
    else:
        return f"""
        You are an expert OCR system for identity documents.
        Analyze the provided image(s) of a "{doc_type}".
        Extract key information and return it as a clean, minified JSON object.
        Do NOT include any explanatory text or markdown formatting.
        The JSON should contain common keys like "firstName", "lastName", "documentNumber", "dateOfBirth", "expiryDate".
        If a field is not present, omit it from the JSON.
        """

@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """Celery task to process documents in the background."""
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Reading images...'})
        image_bytes_list = list(file_contents.values())

        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(image_bytes_list)

        self.update_state(state='PROGRESS', meta={'status': 'Contacting AI for OCR... This may take a moment.'})
        extracted_data = extract_data_with_ollama(image_bytes_list, doc_type)

        if "error" in extracted_data:
            raise Exception(f"AI Processing Error: {extracted_data.get('raw_response', 'Unknown AI error')}")

        self.update_state(state='PROGRESS', meta={'status': 'Saving results to database...'})
        json_data = json.dumps(extracted_data)
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

def extract_data_with_ollama(image_bytes_list, doc_type):
    """Uses the new prompt generation strategy."""
    base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
    
    # Generate the specific prompt for the document type
    prompt = generate_prompt(doc_type)

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llava",
                "prompt": prompt,
                "images": base64_images,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )
        response.raise_for_status()
        response_data = response.json()
        extracted_json_str = response_data.get('response', '{}')
        return json.loads(extracted_json_str)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to communicate with the AI model: {e}"}
    except json.JSONDecodeError as e:
        return {"error": "AI model returned invalid JSON.", "raw_response": extracted_json_str}