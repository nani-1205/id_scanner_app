# app.py

import os
import re
import base64
import traceback
from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load the DocTR OCR Model ---
print("Loading DocTR OCR model...")
predictor = ocr_predictor(pretrained=True, detect_orientation=True, resolve_blocks=False)
print("DocTR model loaded successfully.")

# --- MONGODB CONNECTION ---
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

# ==============================================================================
# --- NEW ADVANCED PARSING HELPER FUNCTIONS ---
# ==============================================================================

def find_word(text, all_words):
    """Finds the first word object that contains the given text."""
    text_lower = text.lower()
    for word in all_words:
        if text_lower in word.value.lower():
            return word
    return None

def find_value_below_key(key_word, all_words):
    """Finds the value located directly below a key word."""
    if not key_word:
        return ""
    
    key_box = key_word.geometry
    # Look for a value in the vertical space below the key
    # x-coordinates should overlap, y-coordinate should be greater
    value_words = [
        word for word in all_words
        if word.geometry[0][1] > key_box[1][1] and  # word's ymin > key's ymax
           max(key_box[0][0], word.geometry[0][0]) < min(key_box[1][0], word.geometry[1][0]) # Horizontal overlap
    ]
    
    if not value_words:
        return ""
        
    # Sort by vertical position, take the closest one
    value_words.sort(key=lambda x: x.geometry[0][1])
    return value_words[0].value

def find_value_between_keys(current_key_word, next_key_word, all_words):
    """Finds the value located horizontally between two key words."""
    if not current_key_word:
        return ""
        
    key_box = current_key_word.geometry
    key_vertical_center = (key_box[0][1] + key_box[1][1]) / 2
    
    left_boundary = key_box[1][0] # current key's xmax
    # If there's a next key, its position is the right boundary. Otherwise, the boundary is the page edge.
    right_boundary = next_key_word.geometry[0][0] if next_key_word else 1.0

    value_words = []
    for word in all_words:
        word_vertical_center = (word.geometry[0][1] + word.geometry[1][1]) / 2
        # Check if word is on the same line, and between the two keys
        if abs(word_vertical_center - key_vertical_center) < 0.02 and \
           word.geometry[0][0] > left_boundary and \
           word.geometry[1][0] < right_boundary:
            value_words.append(word)
            
    value_words.sort(key=lambda x: x.geometry[0][0])
    return " ".join([w.value for w in value_words])


def process_license_with_doctr(image_path):
    """Processes a document image using DocTR with advanced, layout-aware parsing."""
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    
    page_words = [word for page in result.pages for block in page.blocks for line in block.lines for word in line.words]
    
    # --- Define Keys ---
    # Keys that appear on a single line in columns
    ROW_KEYS = ["Nationality", "Sex", "Date of Birth", "Weight", "Height"]
    # Keys where the value is typically below
    COLUMN_KEYS = {
        "license_number": "License No.",
        "expiration_date": "Expiration Date",
        "address": "Address",
        "blood_type": "Blood Type",
        "eye_color": "Eyes Color",
        "conditions": "Conditions",
        "agency_code": "Agency Code"
    }

    extracted_data = {}

    # --- Process Column-based fields (value below key) ---
    for field_name, key_text in COLUMN_KEYS.items():
        key_word = find_word(key_text, page_words)
        extracted_data[field_name] = find_value_below_key(key_word, page_words)

    # --- Process Row-based fields (value between keys) ---
    for i, key_text in enumerate(ROW_KEYS):
        current_key_word = find_word(key_text, page_words)
        next_key_word = None
        if i + 1 < len(ROW_KEYS): # If it's not the last key in the row
            next_key_word = find_word(ROW_KEYS[i+1], page_words)
        
        # Sanitize field name for JSON (e.g., "Date of Birth" -> "date_of_birth")
        field_name = key_text.lower().replace(" ", "_")
        extracted_data[field_name] = find_value_between_keys(current_key_word, next_key_word, all_words=page_words)

    # --- Custom Positional Logic for Full Name ---
    # Find the name by looking for text in a specific region (e.g., above "Nationality")
    nationality_word = find_word("Nationality", page_words)
    if nationality_word:
        nat_box = nationality_word.geometry
        # Find words that are above the nationality line and roughly in the middle of the card
        name_words = [
            word for word in page_words
            if word.geometry[1][1] < nat_box[0][1] and # word's ymax is above nationality's ymin
               word.geometry[0][0] > 0.3 and word.geometry[1][0] < 0.9 # word is horizontally centered
        ]
        name_words.sort(key=lambda x: (x.geometry[0][1], x.geometry[0][0])) # Sort by line, then by x-pos
        extracted_data["full_name"] = " ".join([w.value for w in name_words])
    else:
        extracted_data["full_name"] = "NOT_FOUND"

    # Photo, signature and barcode are not handled by DocTR's text recognition
    extracted_data["photograph_base64"] = "NOT_IMPLEMENTED"
    extracted_data["signature_base64"] = "NOT_IMPLEMENTED"
    extracted_data["barcode_data"] = "NOT_IMPLEMENTED"

    return extracted_data


# ==============================================================================
# --- Flask Routes (No changes needed here) ---
# ==============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_license():
    if 'front_image' not in request.files:
        return jsonify({"error": "Missing front image file"}), 400

    front_file = request.files['front_image']
    if front_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if front_file and allowed_file(front_file.filename):
        front_filename = secure_filename(front_file.filename)
        front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        front_file.save(front_filepath)

        try:
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
    app.run(host='0.0.0.0', debug=True, port=5001)