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
# --- FINAL & HIGH-PRECISION PARSING LOGIC ---
# ==============================================================================

def find_key_phrase(phrase, all_words):
    """Finds a sequence of word objects matching a phrase, ignoring punctuation."""
    phrase_words = [re.sub(r'[^\w]', '', word).lower() for word in phrase.split()]
    if not phrase_words: return None
    for i in range(len(all_words) - len(phrase_words) + 1):
        sequence = all_words[i : i + len(phrase_words)]
        sequence_text = [re.sub(r'[^\w]', '', word.value).lower() for word in sequence]
        if sequence_text == phrase_words:
            first_box, last_box = sequence[0].geometry, sequence[-1].geometry
            if abs(first_box[0][1] - last_box[0][1]) < 0.02: # Check if on the same line
                return sequence
    return None

def get_phrase_bbox(phrase_words):
    """Gets a single bounding box covering a list of word objects."""
    if not phrase_words: return None
    x_min = min(word.geometry[0][0] for word in phrase_words)
    y_min = min(word.geometry[0][1] for word in phrase_words)
    x_max = max(word.geometry[1][0] for word in phrase_words)
    y_max = max(word.geometry[1][1] for word in phrase_words)
    return (x_min, y_min, x_max, y_max)

def get_text_in_area(bbox, all_words):
    """Gets all text within a specified bounding box, sorted left-to-right."""
    if not bbox: return ""
    x_min, y_min, x_max, y_max = bbox
    value_words = [
        word for word in all_words
        if (word.geometry[0][0] + word.geometry[1][0]) / 2 >= x_min and
           (word.geometry[0][0] + word.geometry[1][0]) / 2 <= x_max and
           (word.geometry[0][1] + word.geometry[1][1]) / 2 >= y_min and
           (word.geometry[0][1] + word.geometry[1][1]) / 2 <= y_max
    ]
    value_words.sort(key=lambda x: x.geometry[0][0])
    return " ".join([word.value for word in value_words]).strip(":,.")

def process_license_with_doctr(image_path):
    """Processes a license using a highly specific, layout-aware parsing strategy."""
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    page_words = [word for page in result.pages for block in page.blocks for line in block.lines for word in line.words]
    
    extracted_data = {}

    # Define all keys we expect to find on the license
    KEYS = {
        "name_header": "Last Name, First Name, Middle Name",
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
        "dl_codes": "DL Codes",
        "eye_color": "Eyes Color",
        "conditions": "Conditions",
    }
    
    # Pre-compute the locations of all keys to build a map of the document
    key_bboxes = {name: get_phrase_bbox(find_key_phrase(text, page_words)) for name, text in KEYS.items()}

    # --- Extract values based on their precise, relative locations ---

    # 1. Name: Below the "Last Name..." header, but above the "Nationality" key.
    if key_bboxes["name_header"] and key_bboxes["nationality"]:
        search_area = (key_bboxes["name_header"][0], key_bboxes["name_header"][3], key_bboxes["name_header"][2], key_bboxes["nationality"][1])
        extracted_data["full_name"] = get_text_in_area(search_area, page_words)
    else:
        extracted_data["full_name"] = "NOT_FOUND"

    # 2. Row fields: Nationality, Sex, DOB, Weight, Height
    row_keys = ["nationality", "sex", "date_of_birth", "weight", "height"]
    for i, key_name in enumerate(row_keys):
        if key_bboxes[key_name]:
            right_boundary = 1.0
            if i + 1 < len(row_keys) and key_bboxes[row_keys[i+1]]:
                right_boundary = key_bboxes[row_keys[i+1]][0] # Start of next key
            search_area = (key_bboxes[key_name][2], key_bboxes[key_name][1], right_boundary, key_bboxes[key_name][3])
            extracted_data[key_name] = get_text_in_area(search_area, page_words)
        else:
            extracted_data[key_name] = ""
            
    # 3. Column fields: License No, Expiration Date, Agency Code
    for key_name in ["license_number", "expiration_date", "agency_code"]:
        if key_bboxes[key_name] and key_bboxes["address"]:
             # The value is in the vertical column defined by the key's width, bounded below by the address line
            search_area = (key_bboxes[key_name][0], key_bboxes[key_name][3], key_bboxes[key_name][2], key_bboxes["address"][1])
            extracted_data[key_name] = get_text_in_area(search_area, page_words)
        else:
            extracted_data[key_name] = ""
            
    # 4. Address: It's below the Address key and above the Blood Type key
    if key_bboxes["address"] and key_bboxes["blood_type"]:
        search_area = (key_bboxes["address"][0], key_bboxes["address"][3], 0.95, key_bboxes["blood_type"][1])
        extracted_data["address"] = get_text_in_area(search_area, page_words)
    else:
        extracted_data["address"] = ""

    # 5. Stacked fields: Blood Type/DL Codes and Eyes Color/Conditions
    # We define the space between two keys as the value area.
    if key_bboxes["blood_type"] and key_bboxes["dl_codes"]:
        search_area = (key_bboxes["blood_type"][0], key_bboxes["blood_type"][3], key_bboxes["dl_codes"][2], key_bboxes["dl_codes"][1])
        extracted_data["blood_type_value"] = get_text_in_area(search_area, page_words) # Using temp key
        extracted_data["dl_codes_value"] = get_text_in_area((key_bboxes["dl_codes"][0], key_bboxes["dl_codes"][3], key_bboxes["dl_codes"][2], 1.0), page_words)
    
    if key_bboxes["eye_color"] and key_bboxes["conditions"]:
        search_area = (key_bboxes["eye_color"][0], key_bboxes["eye_color"][3], key_bboxes["conditions"][2], key_bboxes["conditions"][1])
        extracted_data["eye_color_value"] = get_text_in_area(search_area, page_words)
        extracted_data["conditions_value"] = get_text_in_area((key_bboxes["conditions"][0], key_bboxes["conditions"][3], key_bboxes["conditions"][2], 1.0), page_words)
    
    # Re-map the stacked values to their final keys
    extracted_data["blood_type"] = extracted_data.get("blood_type_value", "")
    extracted_data["dl_codes"] = extracted_data.get("dl_codes_value", "")
    extracted_data["eye_color"] = extracted_data.get("eye_color_value", "")
    extracted_data["conditions"] = extracted_data.get("conditions_value", "")

    # Clean up temporary keys
    for k in ["blood_type_value", "dl_codes_value", "eye_color_value", "conditions_value"]:
        extracted_data.pop(k, None)

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