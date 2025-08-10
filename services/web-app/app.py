# services/web-app/app.py
import os
import pika
import json
import uuid
import base64
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
from urllib.parse import quote_plus
from bson import ObjectId

app = Flask(__name__)
# The secret key is needed for session management with Flask-SocketIO
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_super_secret_key_for_development')
socketio = SocketIO(app, async_mode='eventlet')

# --- MongoDB Connection ---
def get_mongo_client():
    mongo_user = os.getenv('MONGO_USER')
    mongo_pass = os.getenv('MONGO_PASS')
    mongo_db_name = os.getenv('MONGO_DB')
    safe_user = quote_plus(mongo_user)
    safe_pass = quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    return client[mongo_db_name]

@app.route('/')
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
    document_type = request.form.get('document_type', 'Uncategorized')
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
        return jsonify({"status": "success", "job_id": job_id})
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500

# --- NEW: Internal endpoint for the worker to post results ---
@app.route('/notify_completion', methods=['POST'])
def notify_completion():
    data = request.json
    job_id = data.get('job_id')
    result = data.get('result')
    if job_id and result:
        # Broadcast the result to the specific client listening for this job_id
        socketio.emit(f'extraction_complete_{job_id}', result)
        return jsonify({"status": "notification sent"})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

# --- NEW: Endpoint to fetch history ---
@app.route('/history', methods=['GET'])
def get_history():
    try:
        db = get_mongo_client()
        # Sort by _id descending to get the most recent documents first, limit to 20
        history_cursor = db.processed_documents.find({}, {'extracted_data': 0}).sort('_id', -1).limit(20)
        
        history = []
        for doc in history_cursor:
            doc['_id'] = str(doc['_id']) # Convert ObjectId to string for JSON
            history.append(doc)
            
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

# --- NEW: Endpoint to fetch a single history item's full data ---
@app.route('/history/<doc_id>', methods=['GET'])
def get_history_item(doc_id):
    try:
        db = get_mongo_client()
        item = db.processed_documents.find_one({'_id': ObjectId(doc_id)})
        if item:
            item['_id'] = str(item['_id'])
            return jsonify(item)
        return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

if __name__ == '__main__':
    # Use socketio.run() instead of app.run()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)