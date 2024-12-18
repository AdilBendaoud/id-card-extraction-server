import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
import pytesseract
import datetime
import re

app = Flask(__name__)

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Updated standard card dimensions (for normalization)
STANDARD_WIDTH = 515
STANDARD_HEIGHT = 321

# Updated region mappings (relative coordinates as percentages)
REGIONS = {
    "first_name_fr": (178 / 515, 70 / 321, 301 / 515, 95 / 321),
    "last_name_fr": (181 / 515, 102 / 321, 290 / 515, 127 / 321),
    "birth_date": (295 / 515, 117 / 321, 381 / 515, 144 / 321),
    "birth_place_fr": (185 / 515, 152 / 321, 346 / 515, 183 / 321),
    "expiry_date": (352 / 515, 275 / 321, 428 / 515, 309 / 321),
    "card_id": (61 / 515, 282 / 321, 149 / 515, 315 / 321)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Ensure file is an image
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Unsupported file format. Only PNG, JPG, and JPEG are allowed.'}), 400

    # Read the image from the uploaded file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image format.'}), 400
    try:
        # # Detect and crop the card
        # cropped_image = detect_and_crop_card(image)
        # if cropped_image is None:
        #     return jsonify({'error': 'Card not detected.'}), 400

        # Extract text data from the cropped card
        extracted_data = extract_id_card_data(image)

        return jsonify(extracted_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# def detect_and_crop_card(image):
#     # Step 1: Resize the image for processing
#     original_height, original_width = image.shape[:2]
#     scaling_factor = min(1000 / original_width, 1000 / original_height)
#     resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#
#     gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#
#     # Step 2: Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Step 3: Use Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)
#     # Step 4: Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Step 5: Find the largest rectangle-like contour
#     largest_contour = None
#     max_area = 0
#     for contour in contours:
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
#         if len(approx) == 4:  # We're looking for quadrilaterals
#             area = cv2.contourArea(contour)
#             if area > max_area:
#                 max_area = area
#                 largest_contour = approx
#
#     if largest_contour is None:
#         return None  # No card detected
#
#     # Step 6: Adjust back to original size
#     pts = (largest_contour.reshape(4, 2) / scaling_factor).astype("float32")
#
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect
#
#     # Step 7: Calculate width and height of the card
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxWidth = int(max(widthA, widthB))
#     maxHeight = int(max(heightA, heightB))
#
#     # Step 8: Create a top-down view of the card
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]
#     ], dtype="float32")
#
#     # Step 9: Apply perspective transform
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth + 50, maxHeight + 50))
#     return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def preprocess_image(image):
    resized = cv2.resize(image, (STANDARD_WIDTH, STANDARD_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    return gray

def extract_region(image, region):
    h, w = image.shape[:2]
    x_start, y_start, x_end, y_end = region
    x1, y1 = int(x_start * w), int(y_start * h)
    x2, y2 = int(x_end * w), int(y_end * h)
    return image[y1:y2, x1:x2]

def clean_text(text, field_name):
    text = text.strip()
    if field_name in ["first_name_fr", "last_name_fr", "birth_place_fr"]:
        text = re.sub(r'[^A-Za-z\s]', '', text)
    # elif field_name in ["birth_date", "expiry_date"]:
    #     # Validate dates in DD/MM/YYYY format
    #     match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
    #     if match:
    #         date_str = match.group(0)
    #         try:
    #             # Parse the date string with the correct format specifier
    #             date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
    #
    #             # Format the valid date in the desired DD.MM.YYYY format
    #             text = date_obj.strftime('%d.%m.%Y')
    #         except ValueError:
    #             text = "Invalid Date"  # If invalid, return error message
    #     else:
    #         text = "Invalid Date"
    elif field_name == "card_id":
        text = re.sub(r'[^A-Za-z0-9]', '', text)
    return text.strip()

def extract_id_card_data(image):
    processed_image = preprocess_image(image)
    data = {}

    for field, region in REGIONS.items():
        region_image = extract_region(processed_image, region)
        try:
            text = pytesseract.image_to_string(region_image, lang='eng', config='--psm 6')
            data[field] = clean_text(text, field)
        except Exception as e:
            data[field] = f"Error: {e}"
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)