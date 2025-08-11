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
# <<< THE CHANGE IS HERE >>>
# This name now matches the new model we are pulling in the entrypoint.sh script.
AI_MODEL = "minicpm"

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")


def extract_text_with_paddleocr(ordered_image_bytes):
    """
    Step 1: Use PaddleOCR for high-accuracy raw text extraction from an ordered list of images.
    """
    full_text = ""
    for i, img_bytes in enumerate(ordered_image_bytes):
        separator = f"\n--- TEXT FROM IMAGE {i+1} ---\n"
        full_text += separator
        try:
            result = paddle_ocr.ocr(img_bytes)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                full_text += "\n".join(texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text

def structure_data_with_specialist_llm(raw_text, base64_images):
    """
    Step 2: Use a powerful multimodal LLM with the user-provided specialist prompt
    to perform final correction and structuring.
    """
    
    # This high-quality prompt works perfectly with MiniCPM-V.
    prompt = f"""
    You are an OCR engine specialized in identity documents (passports, visas, driving licenses). Analyze the input image(s) (front and/or back) and the raw OCR text to extract structured data. Extract the following fields (use English labels and standard formats):

    - document_type: Type of document ("passport", "visa", or "driving_license").
    - full_name: Full name of the holder.
    - date_of_birth: Date of birth in YYYY-MM-DD format.
    - nationality: Nationality as an ISO 3166 country code (e.g. "USA").
    - gender: Gender as a single letter ("M", "F", or other standard code).
    - document_number: Official document number.
    - date_of_issue: Date of issue in YYYY-MM-DD format.
    - expiry_date: Expiration date in YYYY-MM-DD format.
    - issuing_country: Issuing country as ISO code, or issuing_authority: Issuing authority name.
    - photo_present: Boolean true/false indicating if a photo is present.
    - signature_present: Boolean true/false indicating if a signature is present.
    - mrz: (if present) the full Machine Readable Zone string from the document.
    - (If document_type is "visa": include fields visa_type and visa_class.)
    - (If document_type is "driving_license": include field vehicle_classes, a list of categories.)

    Handle any language or script automatically. Use ISO codes and English field names even if the document uses another language. Combine name parts into one string for full_name. Output **only** the specified fields in JSON with exactly these keys. If a field cannot be read, set its value to null or an empty string. Do not invent or guess data.

    --- Raw OCR Text (for guidance, verify against images) ---
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
    """
    The main Celery task orchestrating the universal PaddleOCR -> LLM pipeline.
    """
    try:
        ordered_image_bytes = [file_contents['front']]
        if 'back' in file_contents:
            ordered_image_bytes.append(file_contents['back'])

        # --- Step 1: High-Accuracy OCR ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(ordered_image_bytes)
        if not raw_text.strip():
            raise Exception("PaddleOCR failed to extract any text from the document.")

        # --- Step 2: Intelligent Correction & Structuring ---
        self.update_state(state='PROGRESS', meta={'status': f'AI is analyzing and structuring the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in ordered_image_bytes]
        final_data = structure_data_with_specialist_llm(raw_text, base64_images)

        if "error" in final_data:
            raise Exception(final_data["error"])

        # --- Final Steps ---
        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
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