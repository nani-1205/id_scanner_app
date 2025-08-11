import os
import json
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from celery import Celery, Task
from database import init_db, save_processed_document, get_processed_document
from processor import process_documents_task

# Load environment variables from .env file
load_dotenv()

def make_celery(app):
    """
    Configures a Celery instance to work within the Flask application context.
    """
    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.import_name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    return celery_app

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Celery Configuration ---
app.config.update(
    CELERY=dict(
        broker_url="redis://redis:6379/0",
        result_backend="redis://redis:6379/0",
        task_ignore_result=False,
    )
)
celery = make_celery(app)

@app.before_request
def setup():
    """
    Initializes the database table if it hasn't been created yet.
    """
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page for document uploads.
    """
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        front_file = request.files.get('front_image')
        back_file = request.files.get('back_image')

        if not doc_type or not front_file:
            flash('Please select a document type and upload at least a front image.')
            return redirect(request.url)

        file_contents = {'front': front_file.read()}
        if back_file:
            file_contents['back'] = back_file.read()

        task = process_documents_task.delay(file_contents, doc_type)
        return redirect(url_for('processing_page', task_id=task.id))

    return render_template('index.html')

@app.route('/processing/<task_id>')
def processing_page(task_id):
    """
    Displays a "Please wait" page while the background task runs.
    """
    return render_template('processing.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
    """
    API endpoint for the frontend to poll the status of a background task.
    """
    task = process_documents_task.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending... The worker is picking up the job.'}
    elif task.state == 'SUCCESS':
         # <<< THE FIX IS HERE >>>
         # Changed 'task.sate' to the correct 'task.state'.
         response = {
            'state': task.state,
            'status': task.info.get('status', 'Complete!'),
            'result': task.info.get('result')
        }
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': task.info.get('status', '')}
    else: # Task has failed
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return jsonify(response)

@app.route('/results/<int:doc_id>')
def results(doc_id):
    """
    Displays the final results of a successfully processed document.
    """
    document = get_processed_document(doc_id)
    if not document:
        flash('Document not found!', 'error')
        return redirect(url_for('index'))

    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    
    return render_template('results.html', document=document, extracted_data=extracted_data, face_image_b64=face_image_b64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)