import cv2
import numpy as np
import requests
import base64
import json
from celery import shared_task
from database import save_processed_document

OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# We select the more powerful model here
AI_MODEL = "bakllava" 

def generate_key_value_prompt(doc_type):
    """
    Generates a prompt for the AI to find labels and their corresponding values.
    This is more robust than hardcoded regions of interest (ROI).
    """
    if doc_type == "Driving License":
        return """
        You are an AI assistant specialized in extracting information from Philippine Driver's Licenses.
        Analyze the provided images (front and back).
        Identify all fields and their corresponding values.
        Return the result as a single, minified JSON object.
        The keys in the JSON should be the field names as seen on the license (e.g., "License No.", "Last Name, First Name, Middle Name", "Expiration Date").
        The values should be the text you read for that field.
        Do NOT add any commentary or text outside of the JSON object.
        Combine information from both the front and back images into one JSON object.
        """
    else:
        # Generic fallback for other documents
        return f"""
        Analyze the provided image of a {doc_type}.
        Find all field labels and their corresponding text values.
        Return a single JSON object where keys are the field labels and values are the extracted text.
        """

def post_process_data(raw_data):
    """Cleans up and restructures the AI's raw output for consistency."""
    processed = {}
    
    # Create a mapping from possible AI outputs to our desired database keys
    key_map = {
        "last name, first name, middle name": "fullName",
        "nationality": "nationality",
        "sex": "sex",
        "date of birth": "dateOfBirth",
        "weight (kg)": "weightKg",
        "height (m)": "heightM",
        "address": "address",
        "license no.": "licenseNo",
        "expiration date": "expirationDate",
        "agency code": "agencyCode",
        "blood type": "bloodType",
        "eyes color": "eyesColor",
        "dl codes": "dlCodes",
        "conditions": "conditions",
        "serial number": "serialNumber"
    }

    # Normalize keys from raw_data (lowercase, remove punctuation)
    normalized_raw_data = { k.lower().strip().replace('.', ''): v for k, v in raw_data.items() }

    for map_key, db_key in key_map.items():
        processed[db_key] = normalized_raw_data.get(map_key, "")

    # Special handling for the full name
    if "fullName" in processed:
        full_name_str = processed.pop("fullName", "")
        name_parts = [name.strip() for name in full_name_str.split(',')]
        processed['lastName'] = name_parts[0] if len(name_parts) > 0 else ""
        if len(name_parts) > 1:
            first_middle = name_parts[1].split()
            processed['firstName'] = first_middle[0] if len(first_middle) > 0 else ""
            processed['middleName'] = " ".join(first_middle[1:]) if len(first_middle) > 1 else ""
        else:
            processed['firstName'] = ""
            processed['middleName'] = ""

    return processed


@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """Celery task using Key-Value Pair Extraction for high accuracy."""
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Loading images and preparing AI prompt...'})
        image_bytes_list = list(file_contents.values())
        base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
        
        prompt = generate_key_value_prompt(doc_type)

        self.update_state(state='PROGRESS', meta={'status': f'Contacting AI model ({AI_MODEL})... This may take a moment.'})
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "images": base64_images,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )
        response.raise_for_status()
        raw_extracted_data = json.loads(response.json().get('response', '{}'))

        self.update_state(state='PROGRESS', meta={'status': 'Cleaning and structuring AI output...'})
        final_data = post_process_data(raw_extracted_data)

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