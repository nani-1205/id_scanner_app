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
app.secret_key = os.urandom(24) # Used for flashing messages

# --- Celery Configuration ---
# Configures Celery to use Redis as both the message broker (for sending tasks)
# and the result backend (for storing task status and results).
app.config.update(
    CELERY=dict(
        broker_url="redis://redis:6379/0",
        result_backend="redis://redis:6379/0",
        task_ignore_result=False, # We need to store results to check status
    )
)
# Create the Celery instance
celery = make_celery(app)

@app.before_request
def setup():
    """
    A Flask hook that runs before the first request to the application.
    It initializes the database table if it hasn't been created yet.
    """
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page: displays the upload form (GET) and processes
    the form submission (POST).
    """
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        # Get files by their specific names from the updated HTML form
        front_file = request.files.get('front_image')
        back_file = request.files.get('back_image')

        # Basic validation: A document type and a front image are required.
        if not doc_type or not front_file:
            flash('Please select a document type and upload at least a front image.')
            return redirect(request.url)

        # Create a dictionary to pass to the Celery worker.
        # This is an unambiguous way to identify front vs. back images.
        # We read the file contents here to make them serializable for the task queue.
        file_contents = {'front': front_file.read()}
        if back_file:
            file_contents['back'] = back_file.read()

        # Dispatch the processing task to the background worker.
        # .delay() is the shortcut to send a task to the queue.
        task = process_documents_task.delay(file_contents, doc_type)

        # Immediately redirect the user to the "processing" page.
        # The user does not wait for the long OCR task to complete.
        return redirect(url_for('processing_page', task_id=task.id))

    # For a GET request, just render the main upload page.
    return render_template('index.html')

@app.route('/processing/<task_id>')
def processing_page(task_id):
    """
    Displays a "Please wait" page to the user. The JavaScript on this page
    will poll the /status/<task_id> endpoint.
    """
    return render_template('processing.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
    """
    An API endpoint for the frontend to check the status of a background task.
    It returns a JSON object with the current state and status message.
    """
    task = process_documents_task.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending... The worker is picking up the job.'}
    elif task.state == 'SUCCESS':
         response = {
            'state': task.sate,
            'status': task.info.get('status', 'Complete!'),
            'result': task.info.get('result') # The database ID of the processed document
        }
    elif task.state != 'FAILURE':
        # This handles our custom 'PROGRESS' state.
        response = {'state': task.state, 'status': task.info.get('status', '')}
    else: # Task has failed
        # If the task failed, 'task.info' contains the exception object.
        response = {
            'state': task.state,
            'status': str(task.info), # Convert the exception to a string for display
        }
    return jsonify(response)

@app.route('/results/<int:doc_id>')
def results(doc_id):
    """
    Displays the final results of a successfully processed document
    after retrieving the data from the database.
    """
    document = get_processed_document(doc_id)
    if not document:
        flash('Document not found!', 'error')
        return redirect(url_for('index'))

    # Prepare data for display in the template
    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    
    return render_template('results.html', document=document, extracted_data=extracted_data, face_image_b64=face_image_b64)

# This block is only used for local development without Docker/Gunicorn.
# In our setup, Gunicorn runs the 'app' object directly.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)