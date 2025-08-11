import os
import json
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from database import init_db, save_processed_document, get_processed_document
from processor import process_documents

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.before_request
def setup():
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        files = request.files.getlist('document_images')

        if not doc_type or not files or all(f.filename == '' for f in files):
            flash('Please select a document type and at least one image.')
            return redirect(request.url)

        try:
            extracted_data, face_image, original_images = process_documents(files, doc_type)
            
            if extracted_data is None:
                flash(face_image) # Contains error message in this case
                return redirect(request.url)
            
            json_data = json.dumps(extracted_data)
            
            doc_id = save_processed_document(doc_type, json_data, original_images, face_image)
            
            flash('Document processed successfully!', 'success')
            return redirect(url_for('results', doc_id=doc_id))

        except Exception as e:
            flash(f'An unexpected error occurred: {e}', 'error')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/results/<int:doc_id>')
def results(doc_id):
    document = get_processed_document(doc_id)
    if not document:
        return "Document not found", 404

    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    
    return render_template('results.html', document=document, extracted_data=extracted_data, face_image_b64=face_image_b64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)