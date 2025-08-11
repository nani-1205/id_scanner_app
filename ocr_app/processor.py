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
import fitz # PyMuPDF

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Using the user-specified, high-performance model
AI_MODEL = "minicpm-v:8b"

# --- PaddleOCR Initialization ---
# This heavy object is initialized once when the Celery worker starts.
# It will download its own models on the first run, which may take time.
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")

def normalize_image(image_bytes):
    """
    Opens any image, converts it to a standard RGB JPEG format,
    and returns the standardized image bytes. This prevents errors
    from unsupported image formats like WEBP, HEIC, etc.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error normalizing image: {e}")
        # Fallback to original bytes if normalization fails
        return image_bytes

def process_file_input(file_bytes, filename):
    """
    Accepts a file (image or PDF) and returns a list of standardized image bytes.
    If the file is a PDF, it converts each page into an image.
    """
    images_bytes = []
    
    if filename.lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                # Render page to a high-resolution image
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("jpeg")
                images_bytes.append(normalize_image(img_bytes))
            doc.close()
        except Exception as e:
            print(f"Error processing PDF file '{filename}': {e}")
    else:
        # Process as a single image
        images_bytes.append(normalize_image(file_bytes))
        
    return images_bytes

def extract_text_with_paddleocr(ordered_image_bytes):
    """
    Step 1: Use PaddleOCR for high-accuracy raw text extraction from an ordered list of images.
    """
    full_text = ""
    for i, img_bytes in enumerate(ordered_image_bytes):
        # Add a separator to give the LLM context about the image source (front/back, page 1/2, etc.)
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

def structure_data_with_specialist_llm(raw_text, base64_images):
    """
    Step 2: Use a powerful multimodal LLM with the user-provided specialist prompt
    to perform final correction and structuring.
    """
    
    # This is the high-quality, professional prompt you provided.
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
            # <<< THE FIX IS HERE >>>
            # Increased timeout to 600 seconds (10 minutes) to handle slow processing on CPU.
            timeout=600
        )
        response.raise_for_status()
        final_data = json.loads(response.json().get('response', '{}'))
        return final_data
    except Exception as e:
        print(f"Error during LLM structuring: {e}")
        return {"error": f"The language model failed to structure the text. Error: {e}"}


@shared_task(bind=True)
def process_documents_task(self, file_contents_dict, doc_type):
    """
    The main Celery task orchestrating the universal PaddleOCR -> LLM pipeline.
    """
    try:
        all_image_bytes = []
        original_images_to_save = []

        # Process all uploaded files, converting PDFs to images.
        for key in sorted(file_contents_dict.keys()):
            filename, file_bytes = file_contents_dict[key]
            original_images_to_save.append(file_bytes)
            processed_images = process_file_input(file_bytes, filename)
            all_image_bytes.extend(processed_images)

        if not all_image_bytes:
            raise Exception("No valid images could be processed from the provided file(s).")

        # --- Step 1: High-Accuracy OCR ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(all_image_bytes)
        
        # --- Step 2: Intelligent Correction & Structuring ---
        self.update_state(state='PROGRESS', meta={'status': f'AI is analyzing and structuring the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in all_image_bytes]
        final_data = structure_data_with_specialist_llm(raw_text, base64_images)

        if "error" in final_data:
            raise Exception(final_data["error"])

        # --- Final Steps ---
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