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
FACE_CASCADE = cv2.CascadeClassifier('haarcascades_frontalface_default.xml')
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
    for i, img_bytes in enumerate(ordered_image_bytes):
        separator = f"\n--- TEXT FROM IMAGE {i+1} ---\n"
        full_text += separator
        try:
            # The 'cls' argument is removed as it's set during initialization.
            result = paddle_ocr.ocr(img_bytes)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                full_text += "\n".join(texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text

def structure_with_intelligent_llm(raw_text, base64_images, doc_type_hint):
    """
    Step 2: Use a multimodal LLM with a universal "expert" prompt to structure the data.
    This works for ANY document type.
    """
    prompt = f"""
    You are an expert data extraction AI. Your task is to analyze the provided image(s) and the accompanying raw OCR text to create a structured JSON representation of the document.

    INSTRUCTIONS:
    1.  The user has indicated this document might be a '{doc_type_hint}'. Use this as a helpful hint, but rely on the document's actual content for your final analysis.
    2.  Carefully examine the image(s) to understand the document's layout and to verify the OCR text. Correct any mistakes found in the raw text.
    3.  Identify all key pieces of information on the document.
    4.  Create logical, descriptive, camelCase JSON keys for each piece of information (e.g., "documentTitle", "fullName", "idNumber", "issueDate").
    5.  Populate the JSON with the extracted and corrected data.
    6.  Respond ONLY with the single, minified JSON object and nothing else. Do not add any commentary, notes, or markdown.

    --- Raw Text from OCR Engine (Use this as a guide to be verified against the images) ---
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
        # Create a guaranteed-order list: front first, then back if it exists.
        ordered_image_bytes = [file_contents['front']]
        if 'back' in file_contents:
            ordered_image_bytes.append(file_contents['back'])

        # --- Step 1: High-Accuracy OCR ---
        self.update_state(state='PROGRESS', meta={'status': 'Performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(ordered_image_bytes)
        if not raw_text.strip():
            raise Exception("PaddleOCR failed to extract any text from the document.")

        # --- Step 2: Intelligent Correction & Structuring ---
        self.update_state(state='PROGRESS', meta={'status': f'AI is analyzing the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in ordered_image_bytes]
        # We now call our universal structuring function
        final_data = structure_with_intelligent_llm(raw_text, base64_images, doc_type)

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