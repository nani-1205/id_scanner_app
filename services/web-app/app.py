import os
import pika
import json
import uuid
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'front_image' not in request.files and 'back_image' not in request.files:
        return jsonify({"error": "Please upload at least one image."}), 400

    job_id = str(uuid.uuid4())
    message = {"job_id": job_id}

    for image_type in ['front_image', 'back_image']:
        if image_type in request.files:
            file = request.files[image_type]
            if file and file.filename != '':
                image_bytes = file.read()
                message[image_type] = base64.b64encode(image_bytes).decode('utf-8')

    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
        channel = connection.channel()
        channel.queue_declare(queue='ocr_jobs', durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key='ocr_jobs',
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
        
        return jsonify({
            "status": "success",
            "message": f"Your document scan request has been queued. Job ID: {job_id}",
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)