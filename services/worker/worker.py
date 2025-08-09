# services/worker/worker.py
import os
import pika
import json
import requests
import base64
import time
from pymongo import MongoClient
from urllib.parse import quote_plus

# This print statement will execute the moment the container starts, confirming the script is running.
print("--- OCR Worker Service has started successfully. ---")

def get_mongo_client():
    """Establishes a connection to MongoDB using credentials from environment variables."""
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    
    # CRITICAL FIX: Escape the username and password to handle special characters like '@'.
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    
    # This connection string is correct and includes the necessary authSource.
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    
    client = MongoClient(connection_string)
    return client[mongo_db_name]

def process_image_with_llm(image_base64, image_type):
    """Sends an image to the Ollama service and gets structured data."""
    
    # PROMPT IMPROVEMENT: Added an instruction for the model to specify when a field is unreadable.
    prompt = f"""
    You are an expert OCR system for Philippine government documents.
    Analyze the provided image, which is the {image_type} of a Philippine Driver's License.
    Extract the following fields and return the result ONLY as a single, valid JSON object.
    If a field is not present or not applicable for this side of the card, use an empty string "" as its value.
    If you can see a field label but cannot read its value due to image quality (blur, glare, etc.), the value should be "UNREADABLE".

    Fields to extract:
    - "full_name"
    - "license_number"
    - "expiration_date"
    - "date_of_birth"
    - "nationality"
    - "sex"
    - "height_m"
    - "weight_kg"
    - "address"
    - "blood_type"
    - "eye_color"
    - "agency_code"
    - "dl_codes"
    - "conditions"
    - "serial_number"
    - "emergency_contact_name"
    - "emergency_contact_address"
    - "emergency_contact_tel"
    """
    
    try:
        print(f"Sending {image_type} to Ollama for processing...")
        response = requests.post(
            'http://ollama:11434/api/generate',
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "format": "json"
            },
            timeout=300  # 5-minute timeout for potentially slow AI responses
        )
        response.raise_for_status()
        
        print(f"Received response from Ollama for {image_type}.")
        result_json_string = response.json().get('response', '{}')
        
        # --- CRITICAL DEBUGGING STEP: Print the raw AI response ---
        print(f"--- RAW AI RESPONSE FOR {image_type.upper()} ---:\n{result_json_string}\n--------------------")
        
        return json.loads(result_json_string)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": f"Ollama connection error: {e}"}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama response: {e}")
        print(f"Raw response was: {result_json_string}")
        return {"error": "Failed to decode JSON from model."}

def callback(ch, method, properties, body):
    """The function that is called when a message is received from the queue."""
    print("\n[+] Received new job. Processing...")
    message = json.loads(body)
    job_id = message.get("job_id")
    final_result = {"job_id": job_id}

    # Process front image if it exists
    if "front_image" in message:
        final_result.update(process_image_with_llm(message["front_image"], "front side"))
    
    # Process back image if it exists and merge the results
    if "back_image" in message:
        back_result = process_image_with_llm(message["back_image"], "back side")
        for key, value in back_result.items():
            if value and value not in ["", "N/A"]:
                final_result[key] = value
    
    # This try/except block makes the worker resilient to database failures.
    try:
        db = get_mongo_client()
        collection = db.licenses
        collection.update_one(
            {'job_id': job_id},
            {'$set': final_result},
            upsert=True
        )
        print(f"[+] Job {job_id} successfully processed and saved to MongoDB.")
        # Acknowledge the message (remove from queue) ONLY on success
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("[+] Job acknowledged. Waiting for next job...")

    except Exception as e:
        print(f"[!] Error saving job {job_id} to MongoDB: {e}")
        # Do NOT acknowledge the message. RabbitMQ will re-queue it for another try.
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        print("[!] Job NOT acknowledged. It will be re-queued to prevent data loss.")

def main():
    """Main loop to connect to RabbitMQ and start consuming messages."""
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', heartbeat=600))
            channel = connection.channel()
            channel.queue_declare(queue='ocr_jobs', durable=True)
            channel.basic_qos(prefetch_count=1)  # Process one message at a time
            channel.basic_consume(queue='ocr_jobs', on_message_callback=callback)
            print("[*] Worker is running and waiting for messages.")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"[!] Connection to RabbitMQ failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}. Restarting worker...")
            time.sleep(10)

if __name__ == '__main__':
    main()