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
# --- FINAL & MOST PRECISE PARSING FUNCTIONS ---
# ==============================================================================

def find_key_phrase(phrase, all_words):
    """Finds a sequence of word objects matching a phrase, ignoring punctuation."""
    phrase_words = [re.sub(r'[^\w]', '', word).lower() for word in phrase.split()]
    num_phrase_words = len(phrase_words)
    if num_phrase_words == 0: return None

    for i in range(len(all_words) - num_phrase_words + 1):
        sequence = all_words[i : i + num_phrase_words]
        sequence_text = [re.sub(r'[^\w]', '', word.value).lower() for word in sequence]
        
        if sequence_text == phrase_words:
            first_box = sequence[0].geometry
            last_box = sequence[-1].geometry
            if abs(first_box[0][1] - last_box[0][1]) < 0.02:
                return sequence
    return None

def get_phrase_bbox(phrase_words):
    """Gets a single bounding box that covers a list of word objects."""
    if not phrase_words: return None
    x_min = min(word.geometry[0][0] for word in phrase_words)
    y_min = min(word.geometry[0][1] for word in phrase_words)
    x_max = max(word.geometry[1][0] for word in phrase_words)
    y_max = max(word.geometry[1][1] for word in phrase_words)
    return (x_min, y_min, x_max, y_max)

def find_value_directly_below(key_bbox, all_words):
    """Finds the text on the single line directly below a key's bounding box."""
    if not key_bbox: return ""
    # Define a narrow search area below the key
    search_area = (key_bbox[0], key_bbox[3], key_bbox[2], key_bbox[3] + 0.05)
    
    value_words = [
        word for word in all_words
        if (word.geometry[0][1] + word.geometry[1][1]) / 2 > search_area[1] and
           (word.geometry[0][1] + word.geometry[1][1]) / 2 < search_area[3] and
           (word.geometry[0][0] + word.geometry[1][0]) / 2 > search_area[0] and
           (word.geometry[0][0] + word.geometry[1][0]) / 2 < search_area[2]
    ]
    value_words.sort(key=lambda x: x.geometry[0][0])
    return " ".join([word.value for word in value_words])

def find_value_directly_right(key_bbox, all_words, right_boundary_x=1.0):
    """Finds text on the same line as a key, stopping at a boundary."""
    if not key_bbox: return ""
    key_vertical_center = (key_bbox[1] + key_bbox[3]) / 2
    
    value_words = [
        word for word in all_words
        if abs(((word.geometry[0][1] + word.geometry[1][1]) / 2) - key_vertical_center) < 0.02 and
           word.geometry[0][0] > key_bbox[2] and
           word.geometry[1][0] < right_boundary_x
    ]
    value_words.sort(key=lambda x: x.geometry[0][0])
    return " ".join([word.value for word in value_words])


def process_license_with_doctr(image_path):
    """Processes a license using DocTR with final, high-precision parsing logic."""
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    
    page_words = [word for page in result.pages for block in page.blocks for line in block.lines for word in line.words]
    
    extracted_data = {}

    # --- Define Key Phrases exactly as they appear on the license ---
    KEYS = {
        "full_name": "Last Name, First Name, Middle Name",
        "license_number": "License No.",
        "expiration_date": "Expiration Date",
        "agency_code": "Agency Code",
        "nationality": "Nationality",
        "sex": "Sex",
        "date_of_birth": "Date of Birth",
        "weight": "Weight (kg)",
        "height": "Height(m)",
        "address": "Address",
        "blood_type": "Blood Type",
        "dl_codes": "DL Codes", # This is a separate key now
        "eye_color": "Eyes Color",
        "conditions": "Conditions",
    }
    
    # --- Find all keys first ---
    found_keys = {name: find_key_phrase(text, page_words) for name, text in KEYS.items()}
    found_bboxes = {name: get_phrase_bbox(phrase) for name, phrase in found_keys.items()}

    # --- Extract values using precise logic for each field type ---
    
    # Fields with value below
    for name in ["full_name", "license_number", "expiration_date", "agency_code", "address"]:
        extracted_data[name] = find_value_directly_below(found_bboxes.get(name), page_words)
        
    # Fields in a row
    row_field_names = ["nationality", "sex", "date_of_birth", "weight", "height"]
    for i, name in enumerate(row_field_names):
        right_boundary = 1.0
        # Find the start of the next field to set the boundary
        if i + 1 < len(row_field_names):
            next_key_bbox = found_bboxes.get(row_field_names[i+1])
            if next_key_bbox:
                right_boundary = next_key_bbox[0]
        
        extracted_data[name] = find_value_directly_right(found_bboxes.get(name), page_words, right_boundary_x=right_boundary)

    # Special stacked fields
    for name in ["blood_type", "dl_codes", "eye_color", "conditions"]:
         extracted_data[name] = find_value_directly_right(found_bboxes.get(name), page_words)

    # --- Final cleanup and return ---
    for key, value in extracted_data.items():
        extracted_data[key] = value.strip(" :.")

    # Not implemented fields
    extracted_data["barcode_data"] = "NOT_IMPLEMENTED"
    extracted_data["photograph_base64"] = "NOT_IMPLEMENTED"
    extracted_data["signature_base64"] = "NOT_IMPLEMENTED"

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