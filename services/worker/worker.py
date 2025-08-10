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
from pymongo import MongoClient
from urllib.parse import quote_plus
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from mtcnn.mtcnn import MTCNN
from PIL import Image
import io

# Apply a patch to allow asyncio (used by doctr) to run in a sync environment like Pika's callback
nest_asyncio.apply()

print("--- Fully Offline Document Worker with Real-Time Notification SCRIPT LOADED. ---")

# --- Initialize AI Models on Startup for maximum performance ---
print("Loading local DocTR OCR model...")
try:
    ocr_model = ocr_predictor(pretrained=True, detect_orientation=True)
    print("DocTR OCR model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load DocTR model. Error: {e}")
    ocr_model = None

print("Loading local MTCNN Face Detector model...")
try:
    face_detector = MTCNN()
    print("MTCNN Face Detector model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load MTCNN model. Error: {e}")
    face_detector = None

def get_mongo_client():
    """Establishes a connection to MongoDB."""
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster') # Verify connection
    return client[mongo_db_name]

def extract_face_from_image(image_bytes):
    """Detects, crops, and Base64-encodes the most prominent face in an image."""
    if not face_detector:
        print("[!] Face detector model is not available. Skipping face extraction.")
        return None
    try:
        print("Detecting faces in the image...")
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_rgb = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(image_rgb)
        if not faces:
            print("No faces found in the image.")
            return None
        main_face = max(faces, key=lambda face: face['confidence'])
        x, y, width, height = main_face['box']
        padding_y, padding_x = int(height * 0.4), int(width * 0.2)
        y1, y2 = max(0, y - padding_y), min(image_rgb.shape[0], y + height + padding_y)
        x1, x2 = max(0, x - padding_x), min(image_rgb.shape[1], x + width + padding_x)
        cropped_face = image_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(cropped_face)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        print("Face successfully extracted and encoded.")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"An error occurred during face extraction: {e}")
        return None

def extract_dynamic_json(document_text, document_type, instructions):
    """Stage 2: Sends clean text to a local text model to extract a structured JSON."""
    print(f"Sending text for a '{document_type}' to Llama3 for dynamic JSON extraction...")
    prompt = f"""
    You are an expert data extraction AI. Your task is to analyze the text from a document and convert it into a structured JSON object.
    Document Type: {document_type}
    User Instructions: {instructions}
    Document Text:
    ---
    {document_text}
    ---
    Based on the document type and user instructions, identify all relevant key-value pairs, tables, and lists.
    Return ONLY a single, valid JSON object. Do not include any other text or explanations.
    """
    try:
        response = requests.post(
            'http://ollama-text:11434/api/generate',
            json={"model": "llama3", "prompt": prompt, "stream": False, "format": "json"},
            timeout=180
        )
        response.raise_for_status()
        result_json_string = response.json().get('response', '{}')
        print(f"--- Llama3 Dynamic JSON Response ---:\n{result_json_string}\n--------------------")
        return json.loads(result_json_string)
    except Exception as e:
        print(f"Error during Llama3 JSON extraction: {e}")
        return {"error": f"Llama3 extraction failed: {e}"}

def process_document_offline(image_base64, document_type, instructions):
    """Stage 1: Processes an image using the local DocTR model to get clean text."""
    if not ocr_model: return {"error": "DocTR OCR model is not available."}
    print(f"Parsing document text of type '{document_type}' with local DocTR model...")
    try:
        image_bytes = base64.b64decode(image_base64)
        doc = DocumentFile.from_images(image_bytes)
        result = ocr_model(doc)
        clean_text = result.render()
        print(f"--- DocTR Clean Text Output ---\n{clean_text[:1000]}...\n--------------------")
        return extract_dynamic_json(clean_text, document_type, instructions)
    except Exception as e:
        print(f"An error occurred during DocTR processing: {e}")
        traceback.print_exc()
        return {"error": f"DocTR processing failed: {e}"}

def notify_web_app(job_id, result):
    """Sends the final result to the web-app to be broadcasted via WebSocket."""
    try:
        url = "http://web-app:5001/notify_completion"
        requests.post(url, json={"job_id": job_id, "result": result}, timeout=5)
        print(f"[+] Notified web-app of completion for job {job_id}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Could not notify web-app for job {job_id}. Error: {e}")

def callback(ch, method, properties, body):
    print("\n[+] Received new document processing job.")
    message = json.loads(body)
    job_id = message.get("job_id")
    image_base64 = message["image_base64"]
    image_bytes = base64.b64decode(image_base64)

    # --- HYBRID AI PIPELINE ---
    extracted_photo_base64 = extract_face_from_image(image_bytes)
    extracted_data = process_document_offline(
        image_base64, 
        message.get("document_type"), 
        message.get("instructions")
    )
    
    final_result = {
        "job_id": job_id,
        "document_type": message.get("document_type"),
        "user_instructions": message.get("instructions"),
        "extracted_data": extracted_data,
        "extracted_photograph_base64": extracted_photo_base64 or ""
    }
    
    try:
        db = get_mongo_client()
        collection = db.processed_documents
        # Use insert_one and get the inserted_id for the notification
        insert_result = collection.insert_one(final_result)
        doc_id = str(insert_result.inserted_id)
        print(f"[+] Job {job_id} successfully saved to MongoDB with doc_id {doc_id}.")
        
        # Notify the frontend via the web-app with the ID for fetching
        notify_web_app(job_id, {"status": "completed", "doc_id": doc_id})
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("[+] Job acknowledged.")
    except Exception as e:
        # Notify the frontend of the error
        notify_web_app(job_id, {"status": "error", "message": f"Failed to save to database: {e}"})
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