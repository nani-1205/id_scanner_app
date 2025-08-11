import cv2
import numpy as np
import requests
import base64
import json
import io
from celery import shared_task
from database import save_processed_document

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# The user-suggested, powerful multimodal model
AI_MODEL = "phi3:vision"

def get_extraction_prompt(doc_type):
    """
    Selects a highly-structured, specialized prompt based on the document type.
    This guides the vision model to provide accurate, structured JSON output directly.
    """
    # --- PROMPT FOR PHILIPPINE DRIVER'S LICENSE ---
    if doc_type == "Driving License":
        return """
        Analyze the provided front and back images of a Philippine Driver's License.
        Your task is to meticulously extract the information and populate the following JSON structure.
        Leave the value as an empty string "" if a field is not found.
        Respond ONLY with the single, minified JSON object and nothing else.

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
    # --- PROMPT FOR A STANDARD PASSPORT ---
    elif doc_type == "Passport":
        return """
        Analyze the provided image of a passport's biographical data page.
        Your task is to meticulously extract the information and populate the following JSON structure.
        Leave the value as an empty string "" if a field is not found.
        Respond ONLY with the single, minified JSON object and nothing else.

        {
          "type": "", "issuingCountryCode": "", "passportNo": "", "surname": "", "givenNames": "",
          "nationality": "", "dateOfBirth": "YYYY-MM-DD", "sex": "", "placeOfBirth": "",
          "dateOfIssue": "YYYY-MM-DD", "dateOfExpiry": "YYYY-MM-DD", "issuingAuthority": ""
        }
        """
    # --- FALLBACK FOR ANY OTHER DOCUMENT ---
    else:
        return f"""
        Analyze the provided image of a '{doc_type}'.
        Identify all key information and return it as a structured JSON object.
        Do not add any commentary or notes. Respond ONLY with the JSON object.
        """

@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """Celery task using the powerful Phi-3 Vision model in a single step."""
    try:
        image_bytes_list = list(file_contents.values())
        base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]

        # --- Step 1: Select the right prompt for the job ---
        self.update_state(state='PROGRESS', meta={'status': 'Preparing specialized AI prompt...'})
        prompt = get_extraction_prompt(doc_type)

        # --- Step 2: Call the Vision Model ---
        self.update_state(state='PROGRESS', meta={'status': f'Analyzing document with {AI_MODEL}...'})
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "images": base64_images,
                "stream": False,
                "format": "json"  # Ask the model to guarantee a JSON output
            },
            timeout=180
        )
        response.raise_for_status()
        final_data = json.loads(response.json().get('response', '{}'))
        
        if not final_data:
             raise Exception("AI model returned an empty result. The document may be unclear.")

        # --- Final Steps ---
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(image_bytes_list)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(doc_type, json_data, image_bytes_list, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
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