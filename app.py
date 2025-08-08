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
# --- NEWEST & MOST ADVANCED PARSING FUNCTIONS ---
# ==============================================================================

def find_key_phrase(phrase, all_words):
    """
    Finds a sequence of word objects that match a given phrase.
    This is essential because DocTR splits keys like "License No." into multiple words.
    """
    phrase_words = [word.lower() for word in phrase.split()]
    num_phrase_words = len(phrase_words)

    for i in range(len(all_words) - num_phrase_words + 1):
        # Check if the sequence of words from the document matches our key phrase
        sequence = all_words[i : i + num_phrase_words]
        sequence_text = [word.value.lower().strip(".:,") for word in sequence]
        
        if sequence_text == phrase_words:
            # Check if these words are spatially close on the same line
            first_word_geo = sequence[0].geometry
            last_word_geo = sequence[-1].geometry
            # If the vertical distance between the start and end of the phrase is small, it's on one line
            if abs(first_word_geo[0][1] - last_word_geo[0][1]) < 0.01:
                return sequence # Return the list of matched word objects
    return None

def find_value_below_key(key_word_object, all_words):
    """Finds the value located directly below the last word of a key phrase."""
    if not key_word_object:
        return ""
    
    key_box = key_word_object.geometry
    # Find words whose horizontal span overlaps with the key's horizontal span
    # and which are located below the key
    value_words = [
        word for word in all_words
        if word.geometry[0][1] > key_box[1][1] and  # word's ymin > key's ymax
           max(key_box[0][0], word.geometry[0][0]) < min(key_box[1][0], word.geometry[1][0])
    ]
    if not value_words:
        return ""
    value_words.sort(key=lambda x: x.geometry[0][1]) # Sort by y-position
    return value_words[0].value

def find_value_between_keys(current_key_phrase, next_key_phrase, all_words):
    """Finds the value located horizontally between two key phrases."""
    if not current_key_phrase:
        return ""
        
    last_word_of_key = current_key_phrase[-1]
    key_box = last_word_of_key.geometry
    key_vertical_center = (key_box[0][1] + key_box[1][1]) / 2
    
    left_boundary = key_box[1][0] # current key's xmax
    right_boundary = next_key_phrase[0].geometry[0][0] if next_key_phrase else 1.0

    value_words = []
    for word in all_words:
        word_vertical_center = (word.geometry[0][1] + word.geometry[1][1]) / 2
        if abs(word_vertical_center - key_vertical_center) < 0.02 and \
           word.geometry[0][0] > left_boundary and \
           word.geometry[1][0] < right_boundary:
            value_words.append(word)
            
    value_words.sort(key=lambda x: x.geometry[0][0])
    return " ".join([w.value for w in value_words])

def process_license_with_doctr(image_path):
    """Processes a license using DocTR with phrase-based, layout-aware parsing."""
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    
    page_words = [word for page in result.pages for block in page.blocks for line in block.lines for word in line.words]
    
    extracted_data = {}

    # --- Define Key Phrases ---
    ROW_KEYS = ["Nationality", "Sex", "Date of Birth", "Weight", "Height"]
    COLUMN_KEYS = {
        "license_number": "License No",
        "expiration_date": "Expiration Date",
        "address": "Address",
        "blood_type": "Blood Type",
        "eye_color": "Eyes Color",
        "conditions": "Conditions",
        "agency_code": "Agency Code"
    }

    # --- Process Column-based fields (value below key) ---
    for field_name, key_text in COLUMN_KEYS.items():
        key_phrase_words = find_key_phrase(key_text, page_words)
        if key_phrase_words:
            # Use the last word of the phrase as the anchor
            extracted_data[field_name] = find_value_below_key(key_phrase_words[-1], page_words)
        else:
            extracted_data[field_name] = ""

    # --- Process Row-based fields (value between keys) ---
    for i, key_text in enumerate(ROW_KEYS):
        current_key_phrase = find_key_phrase(key_text, page_words)
        next_key_phrase = None
        if i + 1 < len(ROW_KEYS):
            next_key_phrase = find_key_phrase(ROW_KEYS[i+1], page_words)
        
        field_name = key_text.lower().replace(" ", "_")
        extracted_data[field_name] = find_value_between_keys(current_key_phrase, next_key_phrase, all_words=page_words)

    # --- Custom Logic for Full Name (which is below the "Last Name First Name" header) ---
    name_header_phrase = find_key_phrase("Last Name First Name", page_words)
    if name_header_phrase:
        # The name is the full line of text directly below the header
        last_word_of_header = name_header_phrase[-1]
        header_box = last_word_of_header.geometry
        
        # Find all words on the line below the header
        name_words = [
            w for w in page_words
            if w.geometry[0][1] > header_box[1][1] and # ymin is below header's ymax
               abs(w.geometry[0][1] - (header_box[1][1] + 0.04)) < 0.02 # Vertically close to the line below
        ]
        name_words.sort(key=lambda x: x.geometry[0][0])
        extracted_data["full_name"] = " ".join([w.value for w in name_words])
    else:
        extracted_data["full_name"] = "NOT_FOUND"

    # Not implemented fields
    extracted_data["photograph_base64"] = "NOT_IMPLEMENTED"
    extracted_data["signature_base64"] = "NOT_IMPLEMENTED"
    extracted_data["barcode_data"] = "NOT_IMPLEMENTED"

    return extracted_data


# ==============================================================================
# --- Flask Routes (No changes needed) ---
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