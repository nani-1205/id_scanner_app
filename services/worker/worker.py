# services/worker/worker.py
import os
import pika
import json
import requests
import base64
import time
from pymongo import MongoClient

def get_mongo_client():
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    
    client = MongoClient(f"mongodb://{mongo_user}:{mongo_pass}@mongo:27017/")
    return client[mongo_db_name]

def process_image_with_gemini(image_base64, image_type):
    """Sends an image to Ollama and gets the structured data."""
    
    prompt = f"""
    You are an expert OCR system for Philippine government documents.
    Analyze the provided image, which is the {image_type} of a Philippine Driver's License.
    Extract the following fields and return the result ONLY as a single, valid JSON object.
    If a field is not present or not applicable for this side of the card, use an empty string "" as its value.

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
                "format": "json" # This tells Ollama to ensure the output is valid JSON
            },
            timeout=300 # 5 minute timeout for potentially slow models
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        print(f"Received response from Ollama for {image_type}.")
        # The response from Ollama is a JSON string inside a JSON object.
        result_json_string = response.json().get('response', '{}')
        return json.loads(result_json_string)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama response: {e}")
        print(f"Raw response was: {result_json_string}")
        return {"error": "Failed to decode JSON from model."}

def callback(ch, method, properties, body):
    print("\n[+] Received new job. Processing...")
    message = json.loads(body)
    job_id = message.get("job_id")

    final_result = {"job_id": job_id}

    if "front_image" in message:
        front_result = process_image_with_gemini(message["front_image"], "front side")
        final_result.update(front_result)
    
    if "back_image" in message:
        back_result = process_image_with_gemini(message["back_image"], "back side")
        # Merge results, prioritizing non-empty values
        for key, value in back_result.items():
            if value: # If the back side has a value for a key
                final_result[key] = value

    try:
        db = get_mongo_client()
        collection = db.licenses
        # Use update_one with upsert to create or update the record
        collection.update_one(
            {'job_id': job_id},
            {'$set': final_result},
            upsert=True
        )
        print(f"[+] Job {job_id} successfully processed and saved to MongoDB.")
    except Exception as e:
        print(f"[!] Error saving job {job_id} to MongoDB: {e}")

    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("[+] Job acknowledged. Waiting for next job...")

def main():
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', heartbeat=600))
            channel = connection.channel()
            channel.queue_declare(queue='ocr_jobs', durable=True)
            channel.basic_qos(prefetch_count=1) # Process one message at a time
            channel.basic_consume(queue='ocr_jobs', on_message_callback=callback)
            
            print("[*] Worker is running and waiting for messages. To exit press CTRL+C")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            print("[!] Connection to RabbitMQ failed. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}. Restarting worker...")
            time.sleep(10)

if __name__ == '__main__':
    main()