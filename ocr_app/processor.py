import cv2
import numpy as np
import requests
import base64
import json
from paddleocr import PaddleOCR
from celery import shared_task
from database import save_processed_document
from PIL import Image
import io
import fitz  # PyMuPDF
from datetime import datetime

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
AI_MODEL = "minicpm-v:8b"
OCR_CONFIDENCE_THRESHOLD = 0.80

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR for English and Hindi...")
# Added Hindi 'hi' to better handle bilingual documents like Aadhaar/Passports
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en+hi')
print("PaddleOCR Initialized.")


# --- PROMPT TEMPLATE REPOSITORY ---
# A dictionary of highly specialized prompts for each document type.
PROMPT_TEMPLATES = {
    "passport": """
    You are a data extraction expert for passports. Analyze the provided images and the verified OCR text.
    Your task is to populate the following JSON structure precisely. IGNORE all non-English text (like Hindi).
    The 'full_name' should be constructed from 'given_names' and 'surname'.
    Format all dates as YYYY-MM-DD. Format country codes as 3-letter ISO codes.
    Respond ONLY with the single, minified JSON object.

    JSON Structure:
    {{
      "document_type": "passport", "full_name": null, "surname": null, "given_names": null, "passport_number": null,
      "nationality": null, "issuing_country": null, "gender": null, "date_of_birth": null,
      "date_of_issue": null, "expiry_date": null, "place_of_birth": null, "issuing_authority": null,
      "mrz": null, "additional_data": {{}}
    }}

    --- Raw OCR Text (for guidance, verify against images) ---
    {raw_text}
    """,
    # Add other high-quality templates here as needed
    "driving_license": """...""",
    "aadhaar_card": """...""",
    "emirates_id": """...""",
    "fallback": """
    You are a general data extraction expert. Analyze the provided images and OCR text.
    Identify all key-value pairs and return them in a single, minified JSON object.
    --- Raw OCR Text (for guidance, verify against images) ---
    {raw_text}
    """
}


def preprocess_image_for_ocr(image_bytes):
    # ... (This function is unchanged)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast = clahe.apply(gray)
        coords = np.column_stack(np.where(enhanced_contrast > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        (h, w) = enhanced_contrast.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(enhanced_contrast, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        _, buffer = cv2.imencode('.png', deskewed)
        return buffer.tobytes()
    except Exception as e:
        print(f"Could not preprocess image, using original. Error: {e}")
        return image_bytes

def extract_text_with_paddleocr(ordered_image_bytes):
    # ... (This function is unchanged)
    full_text = ""
    for i, img_bytes in enumerate(ordered_image_bytes):
        separator = f"\n--- TEXT FROM PAGE/IMAGE {i+1} ---\n"
        full_text += separator
        try:
            processed_bytes = preprocess_image_for_ocr(img_bytes)
            result = paddle_ocr.ocr(processed_bytes)
            if result and result[0]:
                high_confidence_texts = [
                    line[1][0] for line in result[0] if line[1][1] > OCR_CONFIDENCE_THRESHOLD
                ]
                full_text += "\n".join(high_confidence_texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text

def classify_document(raw_text):
    """AI Step 1: A simple, reliable task to identify the document type."""
    prompt = f"""
    Based on the following text, what type of document is this?
    Respond with ONLY ONE of the following keywords: "passport", "driving_license", "aadhaar_card", "emirates_id", "other".

    --- Text ---
    {raw_text}
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": AI_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        # Clean the response to get only the keyword
        classification = response.json().get('response', 'other').strip().lower().replace('"', '').replace('.', '')
        return classification
    except Exception as e:
        print(f"Error during document classification: {e}")
        return "fallback"

def structure_data_with_llm(raw_text, base64_images, doc_type):
    """AI Step 2: Selects the correct specialized prompt and extracts data."""
    
    # Select the prompt based on the classification result. Default to fallback.
    prompt_template = PROMPT_TEMPLATES.get(doc_type, PROMPT_TEMPLATES["fallback"])
    final_prompt = prompt_template.format(raw_text=raw_text)
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": AI_MODEL, "prompt": final_prompt, "images": base64_images, "stream": False, "format": "json"},
            timeout=600
        )
        response.raise_for_status()
        extracted_data = json.loads(response.json().get('response', '{}'))
        return extracted_data
    except Exception as e:
        return {"error": f"The language model failed to structure the text. Error: {e}"}

def post_process_and_validate(data):
    # ... (This function is unchanged)
    if not isinstance(data, dict): return data
    for key, value in data.items():
        if "date" in key.lower() and isinstance(value, str):
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%B %d, %Y", "%m/%d/%Y"):
                try:
                    data[key] = datetime.strptime(value, fmt).strftime("%Y-MM-DD")
                    break
                except (ValueError, TypeError): continue
    return data

@shared_task(bind=True)
def process_documents_task(self, file_contents_dict, user_selected_doc_type):
    """The main Celery task orchestrating the Classify-then-Extract pipeline."""
    try:
        # ... (File handling logic is unchanged)
        ordered_image_bytes = [file_contents_dict['front']]
        if 'back' in file_contents_dict:
            ordered_image_bytes.append(file_contents_dict['back'])
            
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(ordered_image_bytes)
        if not raw_text.strip():
            raise Exception("OCR engine failed to extract any text.")

        # --- NEW: Two-Step AI Process ---
        self.update_state(state='PROGRESS', meta={'status': 'AI is identifying the document type...'})
        identified_type = classify_document(raw_text)
        
        self.update_state(state='PROGRESS', meta={'status': f'AI identified a "{identified_type}". Extracting structured data...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in ordered_image_bytes]
        structured_data = structure_data_with_llm(raw_text, base64_images, identified_type)

        if "error" in structured_data:
            raise Exception(structured_data["error"])

        self.update_state(state='PROGRESS', meta={'status': 'Validating and formatting final data...'})
        final_data = post_process_and_validate(structured_data)

        # ... (Face detection and saving to DB is unchanged)
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(ordered_image_bytes)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(user_selected_doc_type, json_data, ordered_image_bytes, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        raise e

def detect_and_crop_face(image_bytes_list):
    # ... (This function is unchanged)
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