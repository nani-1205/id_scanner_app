import cv2
import numpy as np
import requests
import base64
import json
from paddleocr import PaddleOCR
from celery import shared_task
from database import save_processed_document

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
AI_MODEL = "llava-phi3"

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")

def extract_text_with_paddleocr(ordered_image_bytes):
    """
    Step 1: Use PaddleOCR for high-accuracy raw text extraction from an ordered list of images.
    """
    full_text = ""
    # The list is guaranteed to have the front image at index 0 and back at index 1 (if it exists)
    for i, img_bytes in enumerate(ordered_image_bytes):
        # Add a separator to give the LLM context about the image source
        separator = f"\n--- FRONT IMAGE TEXT ---\n" if i == 0 else f"\n--- BACK IMAGE TEXT ---\n"
        full_text += separator
        try:
            result = paddle_ocr.ocr(img_bytes, cls=True)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                full_text += "\n".join(texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text

def structure_with_corrective_llm(raw_text, base64_images, doc_type):
    """
    Step 2: Use a multimodal LLM to correct and structure the text from PaddleOCR.
    """
    if doc_type != "Driving License":
        return {"error": "This document type is not yet supported by the corrective AI engine."}

    prompt = f"""
    You are an AI data verification expert. I have provided you with images of a Philippine Driver's License and the raw text extracted by an OCR engine. The text is separated into 'FRONT IMAGE TEXT' and 'BACK IMAGE TEXT'.
    Your task is to use BOTH the images and the raw text to accurately populate the following JSON structure.
    Use the images to verify and correct any OCR mistakes. For example, the Serial Number is on the back.

    Respond ONLY with the single, minified JSON object and nothing else.

    JSON Structure to populate:
    {{
      "lastName": "", "firstName": "", "middleName": "", "nationality": "", "sex": "",
      "dateOfBirth": "YYYY/MM/DD", "weightKg": "", "heightM": "", "address": "",
      "licenseNo": "", "expirationDate": "YYYY/MM/DD", "agencyCode": "",
      "bloodType": "", "eyesColor": "", "dlCodes": "", "conditions": "", "serialNumber": ""
    }}

    --- Raw Text from OCR Engine (Use this as a guide) ---
    {raw_text}
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": AI_MODEL, "prompt": prompt, "images": base64_images, "stream": False, "format": "json"},
            timeout=180
        )
        response.raise_for_status()
        final_data = json.loads(response.json().get('response', '{}'))
        return final_data
    except Exception as e:
        print(f"Error during LLM structuring: {e}")
        return {"error": f"The language model failed to structure the text. Error: {e}"}

@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """Celery task using the advanced PaddleOCR -> LLM pipeline with ordered images."""
    try:
        # Create a guaranteed-order list: front first, then back if it exists.
        ordered_image_bytes = [file_contents['front']]
        if 'back' in file_contents:
            ordered_image_bytes.append(file_contents['back'])

        # --- Step 1: High-Accuracy OCR ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(ordered_image_bytes)
        if not raw_text.strip():
            raise Exception("PaddleOCR failed to extract any text from the document.")

        # --- Step 2: Intelligent Correction ---
        self.update_state(state='PROGRESS', meta={'status': f'AI is correcting and structuring data...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in ordered_image_bytes]
        final_data = structure_with_corrective_llm(raw_text, base64_images, doc_type)

        if "error" in final_data:
            raise Exception(final_data["error"])

        # --- Final Steps ---
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        # Pass all available images for face detection
        face_image_bytes = detect_and_crop_face(list(file_contents.values()))
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(doc_type, json_data, ordered_image_bytes, face_image_bytes)

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