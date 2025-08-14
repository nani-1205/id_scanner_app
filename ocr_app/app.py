import os
import json
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from celery import Celery, Task
from database import init_db, save_processed_document, get_processed_document, get_history
from processor import process_documents_task
from flask_swagger_ui import get_swaggerui_blueprint
from math import ceil

# Load environment variables from the .env file in the root directory
load_dotenv()

def make_celery(app):
    """
    Configures a Celery instance to work within the Flask application context.
    This allows Celery tasks to access Flask extensions and configuration.
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
app.secret_key = os.urandom(24) # Required for flashing messages

# --- Celery Configuration ---
app.config.update(
    CELERY=dict(
        broker_url="redis://redis:6379/0",
        result_backend="redis://redis:6379/0",
        task_ignore_result=False, # We need to store results to check status
    )
)
celery = make_celery(app)

# --- Swagger UI API Documentation Configuration ---
SWAGGER_URL = '/api/docs'
API_URL = '/api/spec' # This points to our Flask route that serves dynamic JSON
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Offline Document OCR API"}
)
app.register_blueprint(swaggerui_blueprint)

# --- Route to dynamically serve the API specification (swagger.json) ---
@app.route('/api/spec')
def api_spec():
    """Dynamically serves the swagger.json file, injecting the correct host/scheme."""
    with open(os.path.join(app.static_folder, 'swagger.json')) as f:
        swagger_spec = json.load(f)
    
    swagger_spec['host'] = request.host
    swagger_spec['schemes'] = [request.scheme]
    swagger_spec['basePath'] = "/"
    
    return jsonify(swagger_spec)

# --- Flask Hooks ---
@app.before_request
def setup():
    """Initializes the database table before the first request."""
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

# --- Main Application Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main UI page for document uploads."""
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        files = request.files.getlist('document_files')
        
        if not doc_type or not files or all(f.filename == '' for f in files):
            flash('Please select a document type and upload at least one file (image or PDF).')
            return redirect(request.url)
        
        # The worker's `process_file_input` will handle images and PDFs
        file_contents_dict = {f"file_{i}": (f.filename, f.read()) for i, f in enumerate(files)}

        # <<< THE FIX IS HERE >>>
        # The extra 'doc_lang' argument has been removed to match the final processor.py
        task = process_documents_task.delay(file_contents_dict, doc_type)
        return redirect(url_for('processing_page', task_id=task.id))

    return render_template('index.html')

# --- API Endpoint ---
@app.route('/api/v1/extract', methods=['POST'])
def api_extract():
    """API endpoint for programmatic document submission."""
    if 'files' not in request.files:
        return jsonify({"error": "No 'files' part in the request"}), 400
        
    files = request.files.getlist('files')
    doc_type = request.form.get('doc_type', 'Unknown')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400
        
    file_contents_dict = {f"file_{i}": (f.filename, f.read()) for i, f in enumerate(files)}

    # <<< THE FIX IS HERE >>>
    # The extra 'doc_lang' argument has been removed from the call.
    task = process_documents_task.delay(file_contents_dict, doc_type)
    
    return jsonify({
        "message": "Processing started.",
        "task_id": task.id,
        "status_url": url_for('task_status', task_id=task.id, _external=True)
    }), 202

# --- History Page ---
@app.route('/history')
def history():
    """Displays a paginated list of previously processed documents."""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    history_items, total_count = get_history(page, per_page)
    last_page = ceil(total_count / per_page) if total_count > 0 else 1
    
    return render_template('history.html', 
                           history=history_items, 
                           page=page, 
                           per_page=per_page,
                           last_page=last_page)

# --- Task Status & Results Pages ---
@app.route('/processing/<task_id>')
def processing_page(task_id):
    """Displays the 'processing' loading page."""
    return render_template('processing.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
    """API endpoint to check the status of a Celery task."""
    task = process_documents_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'status': task.info.get('status', ''), 'result': task.info.get('result')}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': task.info.get('status', '')}
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

@app.route('/results/<int:doc_id>')
def results(doc_id):
    """Displays the extracted data for a specific document ID."""
    document = get_processed_document(doc_id)
    if not document:
        flash('Document not found!', 'error')
        return redirect(url_for('index'))
    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    return render_template('results.html', document=document, extracted_data=extracted_data, face_image_b64=face_image_b64)

# This block allows running the app with 'python app.py' for local debugging,
# but it is not used by Gunicorn in the Docker container.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)