# services/web-app/app.py
import os
import pika
import json
import uuid
import base64
from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from urllib.parse import quote_plus

app = Flask(__name__)

# --- NEW: MongoDB Connection for the Web App ---
def get_mongo_client():
    """Establishes a connection to MongoDB to fetch results."""
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    return client[mongo_db_name]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'document_image' not in request.files:
        return jsonify({"error": "No document image provided."}), 400

    file = request.files['document_image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    job_id = str(uuid.uuid4())
    
    image_bytes = file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    document_type = request.form.get('document_type', 'Unknown Document')
    instructions = request.form.get('instructions', 'Extract all key-value pairs.')

    message = {
        "job_id": job_id,
        "image_base64": image_base64,
        "document_type": document_type,
        "instructions": instructions
    }

    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
        channel = connection.channel()
        channel.queue_declare(queue='doc_proc_jobs', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='doc_proc_jobs',
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
        
        return jsonify({
            "status": "success",
            "message": f"Job queued successfully. Now polling for results.",
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500

# --- NEW: Endpoint for the Frontend to Poll for Results ---
@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        db = get_mongo_client()
        collection = db.processed_documents
        result = collection.find_one({'job_id': job_id})
        
        if result:
            # We don't need to send the internal MongoDB _id to the frontend
            result.pop('_id', None) 
            return jsonify({"status": "completed", "data": result})
        else:
            return jsonify({"status": "pending"})
            
    except Exception as e:
        print(f"Database query error: {e}")
        return jsonify({"status": "error", "message": "Failed to query database."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)