import os
import json
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from celery import Celery, Task
from database import init_db, save_processed_document, get_processed_document, get_history
from processor import process_documents_task # Make sure processor.py is correctly imported
from flask_swagger_ui import get_swaggerui_blueprint
from math import ceil
from PIL import Image # Import Pillow for image manipulation
import io # Import io for in-memory file operations

load_dotenv()

def make_celery(app):
    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery_app = Celery(app.import_name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    return celery_app

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config.update(
    CELERY=dict(
        broker_url="redis://redis:6379/0",
        result_backend="redis://redis:6379/0",
        task_ignore_result=False,
    )
)
celery = make_celery(app)

# --- Swagger UI API Documentation Configuration ---
SWAGGER_URL = '/api/docs'
API_URL = '/api/spec' # This points to our Flask route that serves dynamic JSON
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Offline Document OCR API"} # Name displayed in Swagger UI
)
app.register_blueprint(swaggerui_blueprint)

# --- Route to dynamically serve the API specification (swagger.json) ---
@app.route('/api/spec')
def api_spec():
    """Dynamically serves the swagger.json file, injecting the correct host/scheme."""
    # Read the base swagger.json from static files
    with open(os.path.join(app.static_folder, 'swagger.json')) as f:
        swagger_spec = json.load(f)
    
    # Inject the correct host and scheme for the API calls in Swagger UI
    swagger_spec['host'] = request.host
    swagger_spec['schemes'] = [request.scheme]
    swagger_spec['basePath'] = "/" # Base path for API endpoints
    
    return jsonify(swagger_spec)


@app.before_request
def setup():
    """Initializes the database table before the first request."""
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main UI page for document uploads."""
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        # Get files by their specific names from the updated HTML form
        front_file = request.files.get('front_image')
        back_file = request.files.get('back_image')

        # Basic validation: A document type and a front image are required.
        if not doc_type or not front_file:
            flash('Please select a document type and upload at least a front image.')
            return redirect(request.url)

        # --- NEW LOGIC: Convert images to a single PDF in memory ---
        pdf_buffer = io.BytesIO() # Create an in-memory buffer for the PDF
        try:
            # Open the front image using Pillow and convert to RGB (standard for PDF)
            front_img_pil = Image.open(io.BytesIO(front_file.read())).convert("RGB")
            
            if back_file:
                # Open and convert the back image if provided
                back_img_pil = Image.open(io.BytesIO(back_file.read())).convert("RGB")
                # Save both images as a multi-page PDF
                front_img_pil.save(pdf_buffer, format='PDF', save_all=True, append_images=[back_img_pil])
            else:
                # Save just the front image as a single-page PDF
                front_img_pil.save(pdf_buffer, format='PDF')
                
        except Exception as e:
            flash(f'Error converting image(s) to PDF: {e}', 'error')
            return redirect(request.url)
        
        pdf_bytes = pdf_buffer.getvalue() # Get the bytes of the created PDF
        # --- END NEW LOGIC ---

        # Pass the single combined PDF file to the Celery worker.
        # Use a generic filename for the PDF as the original names are now irrelevant.
        # The worker's process_file_input function will handle converting this PDF to images.
        file_contents_dict = {'document': ('combined_document.pdf', pdf_bytes)}

        task = process_documents_task.delay(file_contents_dict, doc_type)
        return redirect(url_for('processing_page', task_id=task.id))

    return render_template('index.html')

# --- API ENDPOINT ---
@app.route('/api/v1/extract', methods=['POST'])
def api_extract():
    """
    API endpoint for programmatic document submission.
    Expects a multipart/form-data request with 'doc_type' and 'files'.
    It also converts multiple input images into a single PDF before processing.
    """
    if 'files' not in request.files:
        return jsonify({"error": "No 'files' part in the request"}), 400
        
    files = request.files.getlist('files')
    doc_type = request.form.get('doc_type', 'Unknown') # doc_type is optional for API users
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400
        
    # --- Convert API input images to a single PDF ---
    pdf_buffer = io.BytesIO()
    try:
        first_image_pil = None
        appended_images_pil = []
        
        # Process each uploaded file
        for i, file_storage in enumerate(files):
            # Ensure the file is an image, convert and append
            img_pil = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
            if i == 0:
                first_image_pil = img_pil
            else:
                appended_images_pil.append(img_pil)
        
        if first_image_pil:
            first_image_pil.save(pdf_buffer, format='PDF', save_all=True, append_images=appended_images_pil)
        else:
            return jsonify({"error": "No valid images to process."}), 400

    except Exception as e:
        return jsonify({"error": f"Error converting image(s) to PDF: {e}"}), 400
    
    pdf_bytes = pdf_buffer.getvalue()
    file_contents_dict = {'document': ('combined_document.pdf', pdf_bytes)} # Single PDF to worker

    task = process_documents_task.delay(file_contents_dict, doc_type)
    
    return jsonify({
        "message": "Processing started.",
        "task_id": task.id,
        "status_url": url_for('task_status', task_id=task.id, _external=True) # Full URL for API users
    }), 202

# --- HISTORY PAGE ---
@app.route('/history')
def history():
    """Displays a paginated list of previously processed documents."""
    page = request.args.get('page', 1, type=int)
    per_page = 10 # Number of items per page
    history_items, total_count = get_history(page, per_page)
    
    # Calculate total number of pages for pagination links
    last_page = ceil(total_count / per_page) if total_count > 0 else 1
    
    return render_template('history.html', 
                           history=history_items, 
                           page=page, 
                           per_page=per_page,
                           last_page=last_page)

# --- TASK STATUS & RESULTS PAGES ---
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
        response = {'state': task.state, 'status': task.info.get('status', 'Complete!'), 'result': task.info.get('result')}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': task.info.get('status', '')}
    else: # Task has failed
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)