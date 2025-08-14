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
from PIL import Image
import io

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

SWAGGER_URL = '/api/docs'
API_URL = '/api/spec' 
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "OCR AI API"}
)
app.register_blueprint(swaggerui_blueprint)

@app.route('/api/spec')
def api_spec():
    with open(os.path.join(app.static_folder, 'swagger.json')) as f:
        swagger_spec = json.load(f)
    swagger_spec['host'] = request.host
    swagger_spec['schemes'] = [request.scheme]
    swagger_spec['basePath'] = "/"
    return jsonify(swagger_spec)


@app.before_request
def setup():
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        # Get the new language hint, defaulting to 'en' if empty
        doc_lang = request.form.get('doc_lang') or 'en'
        front_file = request.files.get('front_image')
        back_file = request.files.get('back_image')

        if not doc_type or not front_file:
            flash('Please select a document type and upload at least a front image.')
            return redirect(request.url)

        # Convert uploaded images to a single PDF in memory
        pdf_buffer = io.BytesIO()
        try:
            front_img_pil = Image.open(io.BytesIO(front_file.read())).convert("RGB")
            
            if back_file:
                back_img_pil = Image.open(io.BytesIO(back_file.read())).convert("RGB")
                front_img_pil.save(pdf_buffer, format='PDF', save_all=True, append_images=[back_img_pil])
            else:
                front_img_pil.save(pdf_buffer, format='PDF')
                
        except Exception as e:
            flash(f'Error converting image(s) to PDF: {e}', 'error')
            return redirect(request.url)
        
        pdf_bytes = pdf_buffer.getvalue()

        file_contents_dict = {'document': ('combined_document.pdf', pdf_bytes)}

        # Pass the language hint to the Celery worker
        task = process_documents_task.delay(file_contents_dict, doc_type, doc_lang)
        return redirect(url_for('processing_page', task_id=task.id))

    return render_template('index.html')


@app.route('/api/v1/extract', methods=['POST'])
def api_extract():
    if 'files' not in request.files:
        return jsonify({"error": "No 'files' part in the request"}), 400
        
    files = request.files.getlist('files')
    doc_type = request.form.get('doc_type', 'Unknown')
    doc_lang = request.form.get('doc_lang', 'en') # Also accept lang hint in API
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400
        
    pdf_buffer = io.BytesIO()
    try:
        first_image_pil = None
        appended_images_pil = []
        for i, file_storage in enumerate(files):
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
    file_contents_dict = {'document': ('combined_document.pdf', pdf_bytes)}

    task = process_documents_task.delay(file_contents_dict, doc_type, doc_lang)
    
    return jsonify({
        "message": "Processing started.",
        "task_id": task.id,
        "status_url": url_for('task_status', task_id=task.id, _external=True)
    }), 202

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    history_items, total_count = get_history(page, per_page)
    last_page = ceil(total_count / per_page) if total_count > 0 else 1
    return render_template('history.html', 
                           history=history_items, 
                           page=page, 
                           per_page=per_page,
                           last_page=last_page)

@app.route('/processing/<task_id>')
def processing_page(task_id):
    return render_template('processing.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
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
    document = get_processed_document(doc_id)
    if not document:
        flash('Document not found!', 'error')
        return redirect(url_for('index'))
    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    return render_template('results.html', document=document, extracted_data=extracted_data, face_image_b64=face_image_b64)