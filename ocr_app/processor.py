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

# --- NO GLOBAL PADDLEOCR INITIALIZATION ---
# This is done dynamically inside the task for multilingual support.

# --- HELPER FUNCTIONS ---

def normalize_image(image_bytes):
    # ... (code is unchanged)
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return image_bytes

def preprocess_image_for_ocr(image_bytes):
    # ... (code is unchanged)
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

def process_file_input(file_bytes, filename):
    # ... (code is unchanged)
    images_bytes = []
    if filename.lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("jpeg")
                images_bytes.append(normalize_image(img_bytes))
            doc.close()
        except Exception as e:
            print(f"Error processing PDF file '{filename}': {e}")
    else:
        images_bytes.append(normalize_image(file_bytes))
    return images_bytes

def extract_text_with_paddleocr(ordered_image_bytes, lang_hint):
    """
    Step 1: Dynamically initialize PaddleOCR with the correct languages
    and perform high-accuracy text extraction.
    """
    # <<< THE FIX IS HERE (Part 1) >>>
    # The function now correctly accepts the 'lang_hint' parameter.
    lang_string = '+'.join(lang.strip() for lang in lang_hint.split(','))
    print(f"Initializing PaddleOCR for languages: {lang_string}")
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang_string)

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

# ... (structure_data_with_master_prompt is unchanged)

def post_process_and_validate(data):
    # ... (code is unchanged)
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if "date" in key.lower() and isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%d %b %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    data[key] = datetime.strptime(value, fmt).strftime("%Y-MM-DD")
                    break
                except (ValueError, TypeError):
                    continue
    return data

def detect_and_crop_face(image_bytes_list):
    # ... (code is unchanged)
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

# --- MAIN CELERY TASK ---

@shared_task(bind=True)
def process_documents_task(self, file_contents_dict, doc_type, doc_lang):
    """
    The main Celery task, now with dynamic language support for OCR.
    """
    try:
        all_image_bytes = []
        original_images_to_save = []

        for key in sorted(file_contents_dict.keys()):
            filename, file_bytes = file_contents_dict[key]
            original_images_to_save.append(file_bytes)
            processed_images = process_file_input(file_bytes, filename)
            all_image_bytes.extend(processed_images)

        if not all_image_bytes:
            raise Exception("No valid images could be processed from the provided file(s).")
            
        # <<< THE FIX IS HERE (Part 2) >>>
        # The function call is now valid because the definition has been corrected.
        self.update_state(state='PROGRESS', meta={'status': f'Performing OCR for languages: {doc_lang}...'})
        raw_text = extract_text_with_paddleocr(all_image_bytes, doc_lang)
        
        self.update_state(state='PROGRESS', meta={'status': 'AI is analyzing and structuring the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in all_image_bytes]
        structured_data = structure_data_with_master_prompt(raw_text, base64_images)

        if "error" in structured_data:
            raise Exception(structured_data["error"])

        self.update_state(state='PROGRESS', meta={'status': 'Validating and formatting final data...'})
        final_data = post_process_and_validate(structured_data)

        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(all_image_bytes)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(doc_type, json_data, original_images_to_save, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        raise e