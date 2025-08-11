import cv2 # Correctly imported as cv2
import numpy as np
import requests
import base64
import json
from paddleocr import PaddleOCR
from celery import shared_task
from database import save_processed_document
from PIL import Image
import io
import fitz # PyMuPDF

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
# <<< THE FIX IS HERE >>>
# Changed cv. to the correct cv2.
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
AI_MODEL = "minicpm-v:8b"

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")

# ... (normalize_image function is unchanged)
def normalize_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return image_bytes

# ... (process_file_input function is unchanged)
def process_file_input(file_bytes, filename):
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

# ... (extract_text_with_paddleocr is unchanged)
def extract_text_with_paddleocr(ordered_image_bytes):
    full_text = ""
    for i, img_bytes in enumerate(ordered_image_bytes):
        separator = f"\n--- TEXT FROM PAGE/IMAGE {i+1} ---\n"
        full_text += separator
        try:
            result = paddle_ocr.ocr(img_bytes)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                full_text += "\n".join(texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text

# ... (structure_data_with_specialist_llm is unchanged)
def structure_data_with_specialist_llm(raw_text, base64_images, doc_type_hint):
    prompt = f"""
    You are an OCR engine specialized in identity documents... 
    (Full prompt text)
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
        return {"error": f"The language model failed to structure the text. Error: {e}"}

# ... (process_documents_task is unchanged)
@shared_task(bind=True)
def process_documents_task(self, file_contents_dict, doc_type):
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
        
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(all_image_bytes)
        
        self.update_state(state='PROGRESS', meta={'status': f'AI is analyzing the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in all_image_bytes]
        final_data = structure_data_with_specialist_llm(raw_text, base64_images, doc_type)

        if "error" in final_data:
            raise Exception(final_data["error"])

        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(all_image_bytes)
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        doc_id = save_processed_document(doc_type, json_data, original_images_to_save, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        raise e

def detect_and_crop_face(image_bytes_list):
    """Finds a face from any of the provided images."""
    for img_bytes in image_bytes_list:
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            # <<< THE FIX IS HERE >>>
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