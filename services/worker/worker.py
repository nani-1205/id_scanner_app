# services/worker/worker.py
import os
import pika
import json
import requests
import base64
import time
import nest_asyncio
import traceback
import cv2
import numpy as np
import sys
from pymongo import MongoClient
from urllib.parse import quote_plus
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from mtcnn.mtcnn import MTCNN
from PIL import Image
import io
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize

# Apply a patch to allow asyncio to run in a sync environment
nest_asyncio.apply()

print("--- Professional Grade Document Worker SCRIPT LOADED. ---")

# --- Initialize AI Models on Startup ---
try:
    print("Loading local DocTR OCR model...")
    ocr_model = ocr_predictor(pretrained=True, detect_orientation=True)
    print("DocTR OCR model loaded successfully.")

    print("Loading local MTCNN Face Detector model...")
    face_detector = MTCNN()
    print("MTCNN Face Detector model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load a critical AI model. Error: {e}")
    print("Worker will exit and be restarted by Docker.")
    sys.exit(1)

def get_mongo_client():
    """Establishes a connection to MongoDB."""
    mongo_user, mongo_pass, mongo_db_name = os.getenv('MONGO_USER'), os.getenv('MONGO_PASS'), os.getenv('MONGO_DB')
    safe_user, safe_pass = quote_plus(mongo_user), quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster')
    print("[+] MongoDB connection successful.")
    return client[mongo_db_name]

def extract_face_from_image(image_bytes):
    """Detects, crops, and Base64-encodes the most prominent face."""
    try:
        print("Detecting faces...")
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_rgb = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(image_rgb)
        if not faces: return None
        main_face = max(faces, key=lambda face: face['confidence'])
        x, y, w, h = main_face['box']
        pad_y, pad_x = int(h * 0.4), int(w * 0.2)
        y1, y2 = max(0, y - pad_y), min(image_rgb.shape[0], y + h + pad_y)
        x1, x2 = max(0, x - pad_x), min(image_rgb.shape[1], x + w + pad_x)
        cropped = image_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(cropped)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        print("Face successfully extracted.")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return None

def extract_signature_from_image(image_bytes):
    """Detects, crops, and Base64-encodes signatures using computer vision."""
    try:
        print("Detecting signatures...")
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find areas with high contrast, typical of ink on paper
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours of potential signature areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_signatures = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter based on aspect ratio and size (signatures are usually wide)
            if w > h * 2 and 50 < w < 400 and 20 < h < 150:
                 potential_signatures.append((x, y, w, h))

        if not potential_signatures:
            print("No potential signatures found.")
            return None

        # Assume the largest valid contour is the signature
        x, y, w, h = max(potential_signatures, key=lambda item: item[2] * item[3])
        
        # Crop with padding
        pad = 10
        cropped = image[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad)]

        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        print("Signature successfully extracted.")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return None

def extract_dynamic_json_with_coords(words_with_coords, document_type, instructions):
    """Sends a list of words with coordinates to Llama3 to extract a structured JSON."""
    print(f"Sending text with coordinates for a '{document_type}' to Llama3...")
    
    # Format the input for the LLM
    text_input = "\n".join([f'"{word["value"]}" at bbox {word["geometry"]}' for word in words_with_coords])

    prompt = f"""
    You are an expert data extraction AI. From the following text, where each word is provided with its bounding box coordinates [xmin, ymin, xmax, ymax], create a structured JSON object.

    Document Type: {document_type}
    User Instructions: {instructions}

    Text with Coordinates:
    ---
    {text_input}
    ---

    Your task is to identify key-value pairs. For each value you extract, you MUST also return a 'bbox' field containing the combined bounding box of the words that make up that value.
    The final JSON should look like this:
    {{
      "field_name_1": {{ "value": "extracted value", "bbox": [xmin, ymin, xmax, ymax] }},
      "field_name_2": {{ "value": "another value", "bbox": [xmin, ymin, xmax, ymax] }},
      "line_items": [
        {{ "description": {{"value": "Item 1", "bbox": [...]}}, "amount": {{"value": 10.00, "bbox": [...]}} }}
      ]
    }}
    Combine the bounding boxes for multi-word values by taking the minimum of all xmin/ymin and the maximum of all xmax/ymax.
    Return ONLY a single, valid JSON object.
    """
    try:
        response = requests.post(
            'http://ollama-text:11434/api/generate',
            json={"model": "llama3", "prompt": prompt, "stream": False, "format": "json"},
            timeout=180
        )
        response.raise_for_status()
        result_json_string = response.json().get('response', '{}')
        return json.loads(result_json_string)
    except Exception as e:
        print(f"Error during Llama3 JSON extraction: {e}")
        return {"error": f"Llama3 extraction failed: {e}"}

def process_document_offline(image_base64, document_type, instructions):
    """Processes an image using DocTR to get words and coordinates, then Llama3 to structure them."""
    if not ocr_model: return {"error": "DocTR OCR model is not available."}
    try:
        image_bytes = base64.b64decode(image_base64)
        doc = DocumentFile.from_images(image_bytes)
        result = ocr_model(doc)
        
        # Create a flat list of words with their values and geometries
        words_with_coords = []
        for page in result.pages:
            h, w = page.dimensions
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Normalize coordinates to be relative (0.0 to 1.0)
                        xmin, ymin = word.geometry[0]
                        xmax, ymax = word.geometry[1]
                        words_with_coords.append({
                            "value": word.value,
                            "geometry": [round(xmin/w, 4), round(ymin/h, 4), round(xmax/w, 4), round(ymax/h, 4)]
                        })
        
        return extract_dynamic_json_with_coords(words_with_coords, document_type, instructions)
    except Exception as e:
        print(f"An error occurred during DocTR processing: {e}")
        return {"error": f"DocTR processing failed: {e}"}

def notify_web_app(job_id, result):
    """Notifies the web-app that a job is complete."""
    try:
        url = "http://web-app:5001/notify_completion"
        requests.post(url, json={"job_id": job_id, "result": result}, timeout=5)
        print(f"[+] Notified web-app for job {job_id}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Could not notify web-app for job {job_id}. Error: {e}")

def callback(ch, method, properties, body):
    print("\n[+] Received new job.")
    message = json.loads(body)
    job_id = message.get("job_id")
    image_base64 = message["image_base64"]
    image_bytes = base64.b64decode(image_base64)

    # --- HYBRID AI PIPELINE ---
    photo = extract_face_from_image(image_bytes)
    signature = extract_signature_from_image(image_bytes)
    data = process_document_offline(
        image_base64, 
        message.get("document_type"), 
        message.get("instructions")
    )
    
    final_result = {
        "job_id": job_id,
        "user_id": message.get("user_id"),
        "original_filename": message.get("original_filename"),
        "document_type": message.get("document_type"),
        "extracted_data": data,
        "extracted_photograph_base64": photo or "",
        "extracted_signature_base64": signature or ""
    }
    
    try:
        db = get_mongo_client()
        collection = db.processed_documents
        insert_result = collection.insert_one(final_result)
        doc_id = str(insert_result.inserted_id)
        print(f"[+] Job {job_id} saved to MongoDB with doc_id {doc_id}.")
        notify_web_app(job_id, {"status": "completed", "doc_id": doc_id})
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("[+] Job acknowledged.")
    except Exception as e:
        notify_web_app(job_id, {"status": "error", "message": f"DB Error: {e}"})
        print(f"[!] Error saving job {job_id} to MongoDB: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        print("[!] Job NOT acknowledged. Re-queuing.")

def main():
    """Main loop to connect to RabbitMQ and start consuming messages."""
    print("[*] Worker main function started.")
    while True:
        try:
            print("[*] Attempting to connect to RabbitMQ...")
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', blocked_connection_timeout=300))
            print("[+] RabbitMQ connection successful.")
            channel = connection.channel()
            channel.queue_declare(queue='doc_proc_jobs', durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='doc_proc_jobs', on_message_callback=callback)
            print("[*] General-Purpose Worker is running and waiting for documents.")
            channel.start_consuming()
        except Exception as e:
            print(f"[!] Worker crashed: {e}. Restarting...")
            traceback.print_exc()
            time.sleep(10)

if __name__ == '__main__':
    main()