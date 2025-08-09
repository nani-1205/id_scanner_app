# services/worker/worker.py
import os
import pika
import json
import requests
import base64
import time
import nest_asyncio
import traceback
from pymongo import MongoClient
from urllib.parse import quote_plus
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Apply a patch to allow asyncio to run in a sync environment
nest_asyncio.apply()

print("--- Fully Offline Document Worker SCRIPT LOADED. ---")

# --- Stage 1: Initialize the OCR Model (The "Eyes") ---
# This is done once when the worker starts for maximum performance.
print("Loading local DocTR OCR model...")
try:
    ocr_model = ocr_predictor(pretrained=True, detect_orientation=True)
    print("DocTR OCR model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load DocTR model. Error: {e}")
    ocr_model = None


def get_mongo_client():
    """Establishes a connection to MongoDB."""
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string)
    client.admin.command('ismaster')
    print("[+] MongoDB connection successful.")
    return client[mongo_db_name]

def extract_dynamic_json(document_text, document_type, instructions):
    """Stage 2: Sends clean text to a local text model to extract a structured JSON."""
    print(f"Sending clean text for a '{document_type}' to Llama3 for dynamic JSON extraction...")
    
    prompt = f"""
    You are an expert data extraction AI. Your task is to analyze the text from a document and convert it into a structured JSON object.

    Document Type: {document_type}
    User Instructions: {instructions}

    Document Text:
    ---
    {document_text}
    ---

    Based on the document type and user instructions, identify all relevant key-value pairs, tables, and lists.
    - For tables, create a JSON array of objects.
    - For simple key-value pairs, use them directly.
    - Clean up the data and infer correct data types (numbers, dates) where possible.
    - Return ONLY a single, valid JSON object. Do not include any other text or explanations.
    """
    
    try:
        response = requests.post(
            'http://ollama-text:11434/api/generate',
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=180 # 3 minute timeout
        )
        response.raise_for_status()
        result_json_string = response.json().get('response', '{}')
        print(f"--- Llama3 Dynamic JSON Response ---:\n{result_json_string}\n--------------------")
        return json.loads(result_json_string)
    except Exception as e:
        print(f"Error during Llama3 JSON extraction: {e}")
        return {"error": f"Llama3 extraction failed: {e}"}

def process_document_offline(image_base64, document_type, instructions):
    """Processes an image using the local DocTR -> Llama3 pipeline."""
    if not ocr_model:
        return {"error": "DocTR OCR model is not available."}
        
    print(f"Parsing document of type '{document_type}' with local DocTR model...")
    
    try:
        # DocTR can process image bytes directly
        image_bytes = base64.b64decode(image_base64)
        doc = DocumentFile.from_images(image_bytes)
        
        result = ocr_model(doc)
        
        # Assemble the clean text from the DocTR output
        clean_text = result.render()
        print(f"--- DocTR Clean Text Output ---\n{clean_text[:1000]}...\n--------------------")

        # Now, send the clean text to the text model for JSON extraction
        return extract_dynamic_json(clean_text, document_type, instructions)

    except Exception as e:
        print(f"An error occurred during DocTR processing: {e}")
        traceback.print_exc()
        return {"error": f"DocTR processing failed: {e}"}

def callback(ch, method, properties, body):
    print("\n[+] Received new document processing job.")
    message = json.loads(body)
    job_id = message.get("job_id")
    
    final_result = {
        "job_id": job_id,
        "document_type": message.get("document_type"),
        "user_instructions": message.get("instructions"),
    }
    
    extracted_data = process_document_offline(
        message["image_base64"], 
        message["document_type"], 
        message["instructions"]
    )
    
    final_result["extracted_data"] = extracted_data

    try:
        db = get_mongo_client()
        collection = db.processed_documents
        collection.update_one({'job_id': job_id}, {'$set': final_result}, upsert=True)
        print(f"[+] Job {job_id} successfully saved to MongoDB.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("[+] Job acknowledged.")
    except Exception as e:
        print(f"[!] Error saving job {job_id} to MongoDB: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        print("[!] Job NOT acknowledged. Re-queuing.")

def main():
    while True:
        try:
            print("[*] Worker main function started. Waiting for services to be ready...")
            time.sleep(10) # Add a small delay to ensure RabbitMQ is fully up
            
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