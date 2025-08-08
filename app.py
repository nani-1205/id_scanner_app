# app.py

import os
import cv2
import re
import base64
import traceback
from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- NEW: DocTR Imports ---
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- NEW: Load the DocTR OCR Model ---
# This is done once when the application starts for performance.
print("Loading DocTR OCR model... This may take a moment on first run.")
# Set 'detect_orientation=True' to let DocTR handle rotation automatically
predictor = ocr_predictor(pretrained=True, detect_orientation=True)
print("DocTR model loaded successfully.")

# --- MONGODB CONNECTION ---
# (This part remains the same)
try:
    client = MongoClient(
        host=os.getenv('MONGO_IP'),
        port=int(os.getenv('MONGO_PORT')),
        username=os.getenv('MONGO_USER'),
        password=os.getenv('MONGO_PASS'),
        authSource=os.getenv('MONGO_AUTH_DB'),
        serverSelectionTimeoutMS=5000
    )
    client.admin.command('ismaster')
    print("MongoDB connection successful.")
    db = client.id_scanner_db
    license_collection = db.licenses
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    license_collection = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_value_after_key(key_text, all_words):
    """
    Finds the text value that appears after a given key.
    This is a simple implementation assuming the value is on the same line.
    """
    try:
        # Find the key word
        key_word = next(word for word in all_words if key_text.lower() in word.value.lower())
        
        # Get the key's bounding box [xmin, ymin, xmax, ymax]
        key_box = key_word.geometry
        
        # Find all words that are on the same horizontal line and to the right
        value_words = [
            word for word in all_words
            if word.value != key_word.value and
               abs(word.geometry[1] - key_box[1]) < 0.02 and # ymin is similar (on the same line)
               word.geometry[0] > key_box[2] # xmin is to the right of the key's xmax
        ]
        
        if not value_words:
            return ""
            
        # Sort words by their x-coordinate and join them
        value_words.sort(key=lambda x: x.geometry[0])
        return " ".join([word.value for word in value_words])

    except StopIteration:
        # The key was not found
        return ""

def process_license_with_doctr(image_path):
    """
    Processes a document image using DocTR and intelligently parses the fields.
    """
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    
    # We get a list of all words on the page
    page_words = [word for block in result.pages[0].blocks for line in block.lines for word in line.words]

    # --- NEW: Key-Value Pairing Logic ---
    # Define the keys we are looking for in the document
    key_map = {
        "full_name": "Name",
        "license_number": "License No.",
        "expiration_date": "Expiration Date",
        "nationality": "Nationality",
        "date_of_birth": "Date of Birth",
        "weight_kg": "Weight",
        "height_m": "Height",
        "address": "Address",
        "blood_type": "Blood Type",
        "eye_color": "Eyes Color",
        "conditions": "Conditions",
        "agency_code": "Agency Code",
        "serial_number": "Serial Number"
    }
    
    extracted_data = {}
    for field_name, key_text in key_map.items():
        extracted_data[field_name] = find_value_after_key(key_text, page_words)
    
    # We still need a method for the photo and signature, as DocTR focuses on text.
    # For now, we will leave them blank, but a hybrid approach could be used.
    extracted_data["photograph_base64"] = ""
    extracted_data["signature_base64"] = ""
    extracted_data["barcode_data"] = "Not implemented with DocTR yet"

    # Clean up some fields that might have extra text
    if extracted_data.get("license_number"):
        extracted_data["license_number"] = re.sub(r'[^A-Z0-9-]', '', extracted_data["license_number"])

    return extracted_data

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_license():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({"error": "Missing front or back image"}), 400

    front_file = request.files['front_image']
    back_file = request.files['back_image'] # Back image is not used in this simple DocTR version yet

    if front_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if front_file and allowed_file(front_file.filename):
        front_filename = secure_filename(front_file.filename)
        front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        front_file.save(front_filepath)

        try:
            # --- Call the new DocTR processing function ---
            # NOTE: We are only processing the front image for simplicity here.
            extracted_data = process_license_with_doctr(front_filepath)

            if license_collection is not None:
                license_collection.insert_one(extracted_data.copy())
                print("Data inserted into MongoDB successfully.")
            else:
                print("Skipping database insertion, client not connected.")

            return jsonify(extracted_data)

        except Exception as e:
            print("An error occurred during processing:")
            traceback.print_exc()
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
        finally:
            if os.path.exists(front_filepath):
                os.remove(front_filepath)
    
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)