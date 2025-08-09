# services/worker/worker.py
import os
import pika
import json
import requests
import base64
import time
from pymongo import MongoClient
from urllib.parse import quote_plus
from llama_parse import LlamaParse
import nest_asyncio

# Apply a patch to allow asyncio to run in a sync environment
nest_asyncio.apply()

print("--- General-Purpose Document Worker SCRIPT LOADED. ---")

def get_mongo_client():
    """Establishes a connection to MongoDB."""
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string)
    # The ismaster command is cheap and does not require auth. It will raise an exception on connection failure.
    client.admin.command('ismaster')
    print("[+] MongoDB connection successful.")
    return client[mongo_db_name]

def extract_dynamic_json(document_text, document_type, instructions):
    """Sends clean text and user guidance to a text model to extract a dynamic JSON."""
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
        print(f"--- Llama3 Dynamic JSON Response ---:\n{result_json_string}\n--------------------")
        return json.loads(result_json_string)
    except Exception as e:
        print(f"Error during Llama3 JSON extraction: {e}")
        return {"error": f"Llama3 extraction failed: {e}"}

def process_document(image_base64, document_type, instructions):
    """Processes an image using the LlamaParse -> Llama3 pipeline."""
    print(f"Parsing document of type '{document_type}' with LlamaParse...")
    temp_image_path = "/tmp/document_to_process.tmp"
    with open(temp_image_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    try:
        parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="text")
        documents = parser.load_data(temp_image_path)
        if not documents:
            print("LlamaParse returned no text.")
            return {"error": "LlamaParse could not process the document."}
        clean_text = documents[0].text
        print(f"--- LlamaParse Clean Text Output ---\n{clean_text[:500]}...\n--------------------")
        return extract_dynamic_json(clean_text, document_type, instructions)
    except Exception as e:
        print(f"Error during LlamaParse processing: {e}")
        return {"error": f"LlamaParse API error: {e}"}
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def callback(ch, method, properties, body):
    print("\n[+] Received new document processing job.")
    message = json.loads(body)
    job_id = message.get("job_id")
    final_result = {
        "job_id": job_id,
        "document_type": message.get("document_type"),
        "user_instructions": message.get("instructions"),
    }
    extracted_data = process_document(
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
    """Main loop to connect to RabbitMQ and start consuming messages."""
    print("[*] Worker main function started.")
    while True:
        try:
            print("[*] Attempting to connect to RabbitMQ...")
            # Increased timeout for more resilience on startup
            connection_params = pika.ConnectionParameters(host='rabbitmq', blocked_connection_timeout=300)
            connection = pika.BlockingConnection(connection_params)
            print("[+] RabbitMQ connection successful.")
            
            channel = connection.channel()
            channel.queue_declare(queue='doc_proc_jobs', durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='doc_proc_jobs', on_message_callback=callback)
            
            print("[*] Worker is now consuming messages. Waiting for jobs...")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"[!] RabbitMQ connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            print("[!] Restarting worker in 10 seconds...")
            time.sleep(10)

if __name__ == '__main__':
    main()