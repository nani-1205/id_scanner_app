# app.py

import os
import re
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
# --- FINAL, ROBUST PARSING FUNCTIONS ---
# ==============================================================================

def clean_text(text):
    """A helper function to clean recognized text."""
    return re.sub(r'[^A-Z0-9\s/]', '', text.upper()).strip()

def find_key_phrase(phrase, all_words):
    """Finds a sequence of word objects matching a phrase, ignoring punctuation."""
    phrase_words = [clean_text(word) for word in phrase.split()]
    num_phrase_words = len(phrase_words)
    if num_phrase_words == 0:
        return None

    for i in range(len(all_words) - num_phrase_words + 1):
        sequence = all_words[i : i + num_phrase_words]
        sequence_text = [clean_text(word.value) for word in sequence]
        
        if sequence_text == phrase_words:
            # Check for spatial proximity to ensure they form a real phrase
            first_box = sequence[0].geometry
            last_box = sequence[-1].geometry
            if abs(first_box[0][1] - last_box[0][1]) < 0.02: # On the same line
                return sequence
    return None

def get_phrase_bbox(phrase_words):
    """Gets a single bounding box that covers a list of word objects."""
    if not phrase_words:
        return None
    x_min = min(word.geometry[0][0] for word in phrase_words)
    y_min = min(word.geometry[0][1] for word in phrase_words)
    x_max = max(word.geometry[1][0] for word in phrase_words)
    y_max = max(word.geometry[1][1] for word in phrase_words)
    return (x_min, y_min, x_max, y_max)

def find_value_in_area(bbox, all_words):
    """Finds and concatenates all words within a given bounding box."""
    x_min, y_min, x_max, y_max = bbox
    value_words = [
        word for word in all_words
        if (word.geometry[0][0] + word.geometry[1][0]) / 2 > x_min and
           (word.geometry[0][0] + word.geometry[1][0]) / 2 < x_max and
           (word.geometry[0][1] + word.geometry[1][1]) / 2 > y_min and
           (word.geometry[0][1] + word.geometry[1][1]) / 2 < y_max
    ]
    value_words.sort(key=lambda x: x.geometry[0][0]) # Sort left-to-right
    return " ".join([word.value for word in value_words])

def cleanup_final_data(data):
    """Cleans up the final extracted data dictionary."""
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Remove the original key text if it was accidentally captured
            # For example, if "Weight" key captured "Weight (kg) 78"
            if key in ('weight', 'height'):
                value = re.sub(r'\((KG|M)\)', '', value, flags=re.IGNORECASE)
            cleaned_data[key] = value.strip(":,.")
        else:
            cleaned_data[key] = value
    return cleaned_data

def process_license_with_doctr(image_path):
    """Processes a license using DocTR with final, robust parsing logic."""
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

    # --- Process fields where value is BELOW the key ---
    for field_name, key_text in COLUMN_KEYS.items():
        key_phrase = find_key_phrase(key_text, page_words)
        if key_phrase:
            phrase_bbox = get_phrase_bbox(key_phrase)
            # Define a search area directly below the key phrase
            search_area_bbox = (
                phrase_bbox[0], # x_min
                phrase_bbox[3], # y_max of key
                phrase_bbox[2], # x_max
                phrase_bbox[3] + 0.05 # y_max of key + 5% of page height
            )
            extracted_data[field_name] = find_value_in_area(search_area_bbox, page_words)
        else:
            extracted_data[field_name] = ""
    
    # --- Process fields where value is BETWEEN keys in a row ---
    key_phrase_objects = {key: find_key_phrase(key, page_words) for key in ROW_KEYS}
    
    for i, key_text in enumerate(ROW_KEYS):
        current_phrase = key_phrase_objects.get(key_text)
        field_name = key_text.lower().replace(" ", "_")
        
        if not current_phrase:
            extracted_data[field_name] = ""
            continue

        phrase_bbox = get_phrase_bbox(current_phrase)
        
        # Find the next key to define the right boundary
        next_phrase_bbox = None
        if i + 1 < len(ROW_KEYS):
            next_phrase = key_phrase_objects.get(ROW_KEYS[i+1])
            if next_phrase:
                next_phrase_bbox = get_phrase_bbox(next_phrase)

        search_area_bbox = (
            phrase_bbox[2], # x_max of current key
            phrase_bbox[1], # y_min of current key
            next_phrase_bbox[0] if next_phrase_bbox else 1.0, # x_min of next key or page edge
            phrase_bbox[3]  # y_max of current key
        )
        extracted_data[field_name] = find_value_in_area(search_area_bbox, page_words)

    # --- Custom Logic for Full Name ---
    name_header_phrase = find_key_phrase("Last Name, First Name, Middle Name", page_words)
    if name_header_phrase:
        header_bbox = get_phrase_bbox(name_header_phrase)
        search_area_bbox = (header_bbox[0], header_bbox[3], header_bbox[2], header_bbox[3] + 0.05)
        extracted_data["full_name"] = find_value_in_area(search_area_bbox, page_words)
    else:
        extracted_data["full_name"] = "NOT_FOUND"

    # Not implemented fields
    extracted_data["photograph_base64"] = "NOT_IMPLEMENTED"
    extracted_data["signature_base64"] = "NOT_IMPLEMENTED"
    extracted_data["barcode_data"] = "NOT_IMPLEMENTED"

    return cleanup_final_data(extracted_data)


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
    if 'front_image' not in request.files: return jsonify({"error": "Missing front image file"}), 400
    front_file = request.files['front_image']
    if front_file.filename == '': return jsonify({"error": "No selected file"}), 400

    if front_file and allowed_file(front_file.filename):
        front_filename = secure_filename(front_file.filename)
        front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        front_file.save(front_filepath)

        try:
            extracted_data = process_license_with_doctr(front_filepath)
            if license_collection is not None:
                license_collection.insert_one(extracted_data.copy())
                print("Data inserted into MongoDB successfully.")
            return jsonify(extracted_data)
        except Exception as e:
            print("An error occurred during processing:")
            traceback.print_exc()
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
        finally:
            if os.path.exists(front_filepath): os.remove(front_filepath)
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)