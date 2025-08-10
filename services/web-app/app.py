# services/web-app/app.py
import os
import pika
import json
import uuid
import base64
from flask import Flask, request, render_template, jsonify, session
from flask_socketio import SocketIO
from pymongo import MongoClient
from urllib.parse import quote_plus
from bson import ObjectId

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
# Use eventlet as the async_mode for production readiness
socketio = SocketIO(app, async_mode='eventlet')

# --- MongoDB Connection ---
def get_mongo_client():
    mongo_user, mongo_pass, mongo_db_name = os.getenv('MONGO_USER'), os.getenv('MONGO_PASS'), os.getenv('MONGO_DB')
    safe_user, safe_pass = quote_plus(mongo_user), quote_plus(mongo_pass)
    connection_string = f"mongodb://{safe_user}:{safe_pass}@mongo:27017/?authSource={mongo_db_name}"
    client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
    return client[mongo_db_name]

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = 'default_user'
        session['username'] = 'Admin'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'document_image' not in request.files: return jsonify({"error": "No document image provided."}), 400
    file = request.files['document_image']
    if file.filename == '': return jsonify({"error": "No selected file."}), 400

    job_id = str(uuid.uuid4())
    message = {
        "job_id": job_id, "image_base64": base64.b64encode(file.read()).decode('utf-8'),
        "document_type": request.form.get('document_type', 'Uncategorized'),
        "instructions": request.form.get('instructions', 'Extract all key-value pairs.'),
        "user_id": session.get('user_id'), "original_filename": file.filename
    }
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
        channel = connection.channel()
        channel.queue_declare(queue='doc_proc_jobs', durable=True)
        channel.basic_publish(exchange='', routing_key='doc_proc_jobs', body=json.dumps(message), properties=pika.BasicProperties(delivery_mode=2))
        connection.close()
        return jsonify({"status": "success", "job_id": job_id})
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500

@app.route('/notify_completion', methods=['POST'])
def notify_completion():
    data = request.json
    job_id = data.get('job_id')
    result = data.get('result')
    if job_id and result:
        socketio.emit(f'extraction_complete_{job_id}', result)
        return jsonify({"status": "notification sent"})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/history', methods=['GET'])
def get_history():
    user_id = session.get('user_id')
    search_query = request.args.get('q', '')
    query_filter = {'user_id': user_id}
    if search_query: query_filter['$text'] = {'$search': search_query}
    try:
        db = get_mongo_client()
        history_cursor = db.processed_documents.find(
            query_filter, 
            {'extracted_data': 0, 'extracted_photograph_base64': 0, 'extracted_signature_base64': 0}
        ).sort('_id', -1).limit(50)
        history = [{'id': str(doc['_id']), 'filename': doc.get('original_filename', 'N/A'), 'type': doc.get('document_type', 'N/A')} for doc in history_cursor]
        return jsonify(history)
    except Exception as e: return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/document/<doc_id>', methods=['GET', 'PUT'])
def get_document(doc_id):
    db = get_mongo_client()
    user_id = session.get('user_id')
    try:
        if request.method == 'GET':
            item = db.processed_documents.find_one({'_id': ObjectId(doc_id), 'user_id': user_id})
            if item:
                item['_id'] = str(item['_id'])
                return jsonify(item)
            return jsonify({"error": "Document not found"}), 404
        if request.method == 'PUT':
            updates = request.json.get('extracted_data')
            result = db.processed_documents.update_one(
                {'_id': ObjectId(doc_id), 'user_id': user_id}, {'$set': {'extracted_data': updates}}
            )
            if result.matched_count: return jsonify({"status": "success", "message": "Document updated."})
            return jsonify({"error": "Document not found or permission denied"}), 404
    except Exception as e: return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/dashboard_stats', methods=['GET'])
def get_dashboard_stats():
    db = get_mongo_client()
    user_id = session.get('user_id')
    try:
        total_docs = db.processed_documents.count_documents({'user_id': user_id})
        pipeline = [
            {'$match': {'user_id': user_id}}, {'$group': {'_id': '$document_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}, {'$limit': 5}
        ]
        doc_types = list(db.processed_documents.aggregate(pipeline))
        return jsonify({"total_documents": total_docs, "document_types": doc_types})
    except Exception as e: return jsonify({"error": f"Database error: {str(e)}"}), 500

if __name__ == '__main__':
    # This is the corrected line. "0.0.0.0" is a special address that doesn't require DNS lookup.
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)