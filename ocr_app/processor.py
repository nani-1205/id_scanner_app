import cv2
import numpy as np
import requests
import base64
import json
import io
from paddleocr import PaddleOCR
from celery import shared_task
from database import save_processed_document
from PIL import Image # Import the Pillow library

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
AI_MODEL = "minicpm-v:8b" # Using the user-specified, high-performance model

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")

def normalize_image(image_bytes):
    """
    Opens an image, converts it to a standard format (RGB JPEG),
    and returns the standardized image bytes. This prevents errors
    from unsupported image formats like WEBP, HEIC, etc.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB to handle various modes like RGBA, P, etc.
        image = image.convert("RGB")
        
        # Save the image to an in-memory buffer as a JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error normalizing image: {e}")
        # If normalization fails, return the original bytes as a fallback
        return image_bytes

def extract_text_with_paddleocr(ordered_image_bytes):
    # ... (This function remains unchanged)
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
    # ... (This function remains unchanged)
    prompt = f"""
    You are an OCR engine specialized in identity documents...
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
        return {"error": f"The language model failed to structure the text. Error: {e}"}

@shared_task(bind=True)
def process_documents_task(self, file_contents, doc_type):
    """
    The main Celery task orchestrating the final, robust pipeline.
    """
    try:
        # --- NEW: Image Normalization Step ---
        self.update_state(state='PROGRESS', meta={'status': 'Standardizing image formats...'})
        normalized_front = normalize_image(file_contents['front'])
        normalized_back = None
        if 'back' in file_contents:
            normalized_back = normalize_image(file_contents['back'])

        ordered_image_bytes = [normalized_front]
        if normalized_back:
            ordered_image_bytes.append(normalized_back)
        
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
        face_image_bytes = detect_and_crop_face(ordered_image_bytes) # Use normalized images for this too
        
        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)
        # We still save the original, high-quality images to the database
        original_images_to_save = [file_contents['front']]
        if 'back' in file_contents:
            original_images_to_save.append(file_contents['back'])
        doc_id = save_processed_document(doc_type, json_data, original_images_to_save, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        raise e

def detect_and_crop_face(image_bytes_list):
    # ... (This function remains unchanged)
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

# To keep the code block shorter, I've omitted the full prompt text,
# but it should be the same professional prompt from the last step.
def structure_data_with_specialist_llm(raw_text, base64_images):
    prompt = f"""
    You are an OCR engine specialized in identity documents...
    """ # <-- The full prompt goes here
    # ... rest of the function
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