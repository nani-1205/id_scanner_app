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
# Set 'resolve_blocks=False' for faster processing if we only need words
predictor = ocr_predictor(pretrained=True, detect_orientation=True, resolve_blocks=False)
print("DocTR model loaded successfully.")

# --- MONGODB CONNECTION ---
# This part remains the same. Ensure your .env file is correct.
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
    This corrected version handles DocTR's geometry format properly.
    """
    try:
        # Find the key word (case-insensitive search)
        key_word = next(word for word in all_words if key_text.lower() in word.value.lower())
        
        # Get the key's bounding box: ((xmin, ymin), (xmax, ymax))
        key_box = key_word.geometry
        
        # Define the vertical center of the key word to find other words on the same line
        key_vertical_center = (key_box[0][1] + key_box[1][1]) / 2  # (ymin + ymax) / 2
        
        # Find all words that are on roughly the same horizontal line and to the right
        value_words = []
        for word in all_words:
            # Check if the word is part of the key itself to avoid self-matching
            is_part_of_key = any(part.lower() == word.value.lower() for part in key_text.split())
            if not is_part_of_key:
                word_vertical_center = (word.geometry[0][1] + word.geometry[1][1]) / 2
                
                # Check if the word is vertically aligned AND to the right of the key
                # A tolerance of 0.02 (2% of image height) is used for vertical alignment
                if abs(word_vertical_center - key_vertical_center) < 0.02 and \
                   word.geometry[0][0] > key_box[1][0]:  # word's xmin > key's xmax
                    value_words.append(word)

        if not value_words:
            return ""
            
        # Sort words by their x-coordinate and join them
        value_words.sort(key=lambda x: x.geometry[0][0])
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
    
    # We get a flat list of all words on the page for easier processing
    page_words = [word for page in result.pages for block in page.blocks for line in block.lines for word in line.words]

    # --- Key-Value Pairing Logic ---
    # Define the keys we are looking for in the document.
    # The keys should be simple and unique.
    key_map = {
        "license_number": "License No",
        "expiration_date": "Expiration",
        "nationality": "Nationality",
        "date_of_birth": "Birth",
        "weight_kg": "Weight",
        "height_m": "Height",
        "address": "Address",
        "blood_type": "Blood",
        "eye_color": "Eyes",
        "conditions": "Conditions",
        "agency_code": "Agency",
        "serial_number": "Serial" # Using a partial key for robustness
        # Name doesn't have a clear key, would need different logic
    }
    
    extracted_data = {}
    for field_name, key_text in key_map.items():
        extracted_data[field_name] = find_value_after_key(key_text, page_words)
    
    # Special logic for fields without clear keys
    # This is an example of how you might find the name based on position
    # For now, we leave it blank as it's more complex
    extracted_data["full_name"] = "NEEDS_CUSTOM_LOGIC" 
    
    # Photo, signature and barcode are not handled by DocTR's text recognition
    extracted_data["photograph_base64"] = "NOT_IMPLEMENTED"
    extracted_data["signature_base64"] = "NOT_IMPLEMENTED"
    extracted_data["barcode_data"] = "NOT_IMPLEMENTED"

    return extracted_data

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_license():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({"error": "Missing front or back image"}), 400

    front_file = request.files['front_image']
    # The back image is available if you want to extend the logic
    # back_file = request.files['back_image'] 

    if front_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if front_file and allowed_file(front_file.filename):
        front_filename = secure_filename(front_file.filename)
        front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        front_file.save(front_filepath)

        try:
            # --- Call the new DocTR processing function ---
            # NOTE: We are only processing the front image for simplicity.
            # You could easily process the back image as well and merge the results.
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
    # When deploying, use a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', debug=True, port=5001)