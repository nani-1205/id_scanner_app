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
LANGUAGE_MODEL = "llama3"

# --- PROMPT TEMPLATE REPOSITORY ---
# This is the core of our adaptable engine. We define specialized prompts for each document.
DOCUMENT_PROMPTS = {
    "driving license": """
    You are an expert data extraction assistant. The user has provided raw text from a Philippine Driver's License.
    Your task is to parse this text and populate the following JSON structure.
    Fill every field with the corresponding data from the text.
    If data for a field is not present in the text, leave the value as an empty string "".
    Do NOT add any commentary or notes. Respond ONLY with the JSON object.

    JSON Structure:
    {{
      "lastName": "", "firstName": "", "middleName": "", "nationality": "", "sex": "",
      "dateOfBirth": "YYYY/MM/DD", "weightKg": "", "heightM": "", "address": "",
      "licenseNo": "", "expirationDate": "YYYY/MM/DD", "agencyCode": "",
      "bloodType": "", "eyesColor": "", "dlCodes": "", "conditions": "", "serialNumber": ""
    }}

    --- Raw Text from OCR ---
    {raw_text}
    """,
    "passport": """
    You are an expert data extraction assistant. The user has provided raw text from a Passport.
    Your task is to parse this text and populate the following JSON structure.
    Fill every field with the corresponding data from the text.
    If data for a field is not present in the text, leave the value as an empty string "".
    Do NOT add any commentary or notes. Respond ONLY with the JSON object.

    JSON Structure:
    {{
      "type": "", "issuingCountryCode": "", "passportNo": "", "surname": "", "givenNames": "",
      "nationality": "", "dateOfBirth": "YYYY-MM-DD", "sex": "", "placeOfBirth": "",
      "dateOfIssue": "YYYY-MM-DD", "dateOfExpiry": "YYYY-MM-DD", "issuingAuthority": ""
    }}

    --- Raw Text from OCR ---
    {raw_text}
    """,
    "fallback": """
    You are a general-purpose data extraction assistant. The user has provided raw text from an unknown document.
    Analyze the text and identify all key-value pairs you can find (e.g., "Name": "John Doe", "ID Number": "12345").
    Return the result as a single, minified JSON object. Do NOT add any commentary or notes.

    --- Raw Text from OCR ---
    {raw_text}
    """
}

def extract_raw_text_with_tesseract(image_bytes_list):
    """Step 1: Use Tesseract for high-accuracy OCR."""
    full_text = ""
    for image_bytes in image_bytes_list:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang='eng')
            full_text += text + "\n\n"
        except Exception as e:
            print(f"Error during Tesseract OCR: {e}")
    return full_text

def classify_document_with_llama3(raw_text):
    """Step 2: Use Llama3 to identify the type of document."""
    prompt = f"""
    Analyze the following text extracted from a document.
    What type of document is this?
    Please respond with ONLY one of the following choices: "Driving License", "Passport", "Other".

    --- Text from Document ---
    {raw_text}
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": LANGUAGE_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        # Clean the response to get only the classification
        classification = response.json().get('response', 'Other').strip().lower().replace('"', '')
        return classification
    except Exception as e:
        print(f"Error during document classification: {e}")
        return "other" # Default to 'other' on failure

def structure_text_with_llama3(raw_text, doc_type):
    """Step 3: Select the correct specialized prompt and structure the data."""
    
    # Select the prompt based on the classification result. Default to fallback.
    prompt_template = DOCUMENT_PROMPTS.get(doc_type, DOCUMENT_PROMPTS["fallback"])
    final_prompt = prompt_template.format(raw_text=raw_text)
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": LANGUAGE_MODEL, "prompt": final_prompt, "stream": False, "format": "json"},
            timeout=180
        )
        response.raise_for_status()
        response_data = response.json().get('response', '{}')
        return json.loads(response_data)
    except Exception as e:
        print(f"Error structuring text with Llama3: {e}")
        return {"error": f"The language model failed to structure the text. Error: {e}"}

@shared_task(bind=True)
def process_documents_task(self, file_contents, user_selected_doc_type):
    """The main Celery task orchestrating the advanced OCR pipeline."""
    try:
        image_bytes_list = list(file_contents.values())
        
        # --- Step 1: High-Accuracy OCR ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_raw_text_with_tesseract(image_bytes_list)
        if not raw_text.strip():
            raise Exception("OCR engine failed to extract any text from the document.")

        # --- Step 2: AI-Powered Classification ---
        self.update_state(state='PROGRESS', meta={'status': 'AI is identifying the document type...'})
        identified_type = classify_document_with_llama3(raw_text)
        
        # --- Step 3: Specialized, AI-Powered Structuring ---
        self.update_state(state='PROGRESS', meta={'status': f'AI identified a "{identified_type}". Extracting structured data...'})
        final_data = structure_text_with_llama3(raw_text, identified_type)
        if "error" in final_data:
            raise Exception(final_data["error"])

        # --- Final Steps ---
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(image_bytes_list)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        # We save with the type the user selected, but the data is from the AI's identified type
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(user_selected_doc_type, json_data, image_bytes_list, face_image_bytes)

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