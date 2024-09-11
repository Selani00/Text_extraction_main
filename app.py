from flask import Flask, request, jsonify
import tf_keras as keras
from PIL import Image
import cv2
import numpy as np
import pytesseract
from transformers import pipeline
from langdetect import LangDetectException
import io

from utils import labels, boxes,word_lists
from functions import calculate_visibility, clear_image, identify_image_type,save_data_to_excel,correct_word_ignore_case
from text_validation import validate_extracted_text

app = Flask(__name__)

# Initialize Tesseract and the zero-shot classifier
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

@app.route('/extract_text', methods=['POST'])
def extract_text_from_image():
    # Get the image from the request
    image = get_image_from_request(request)

    # Resize the image (in-memory processing)
    resized_image = resize_image(image, (2481, 3507))

    # Crop the image based on outer box
    cropped_image = crop_image(resized_image)

    # Resize the cropped image to the final dimensions
    final_resized_image = resize_image(cropped_image, (2480, 3500))

    # Detect the type of image
    Type = identify_image_type(final_resized_image)

    # Extract text from predefined boxes and collect confidence scores
    detected_texts, accuracy = extract_text_with_accuracy(final_resized_image, Type)

    image_id = detected_texts.get('Image_ID', 'Unknown')

    # Measure visibility percentage of the input image
    visibility = calculate_visibility(final_resized_image)

    # Prepare the JSON response
    response_data = {
        "Accuracy": f"{accuracy:.2f}%",
        "Document Type": "Vehicle Registration Document",
        "ID": image_id, 
        "Visibility": f"{visibility:.2f}%",
        "data": detected_texts,
        "message": "Image Converted successfully.",
        "status": 200
    }

    save_data_to_excel(detected_texts, accuracy, visibility, image_id)

    return jsonify(response_data)


# Function to extract the image from the request and open it in-memory
def get_image_from_request(request):
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    return image


# Function to resize the image in-memory
def resize_image(image, dimension):
    return image.resize(dimension)


# Function to crop the image based on the outer box (in-memory processing)
def crop_image(image):
    # Convert to OpenCV image format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    margin = 60
    y = max(0, y - margin)
    h = h + margin

    # Crop the image in-memory
    cropped_image_cv = image_cv[y:y+h, x:x+w]
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB))
    return cropped_image_pil

# Function to adjust the box dynamically based on the confidence
def adjust_box(x, y, w, h, image_cv, confidence_threshold=50):
    box_confidence_2 = 0

    # First adjustment (initial decrease in x and y)
    new_x = max(0, int(x - 10))
    new_y1 = max(0, int(y - 50))

    # Recalculate the confidence with the new adjusted box
    roi = image_cv[new_y1:new_y1 + h, new_x:new_x + w]
    custom_config = r'--oem 3 --psm 6 -l eng'
    d = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)

    # Extract valid confidence values (ignoring -1)
    valid_confidences = [int(conf) for conf in d['conf'] if int(conf) != -1]
    box_confidence_1 = np.mean(valid_confidences) if valid_confidences else 0
    

    # Check if the box confidence is still below the threshold
    if box_confidence_1 < confidence_threshold:
       
        new_y2 = max(0, int(y + 50))

        # Recalculate confidence with the second adjustment
        roi = image_cv[new_y2:new_y2 + h, new_x:new_x + w]
        d = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract valid confidence values (ignoring -1)
        valid_confidences = [int(conf) for conf in d['conf'] if int(conf) != -1]
        box_confidence_2 = np.mean(valid_confidences) if valid_confidences else 0
        

    if (box_confidence_2 != 0):
        if box_confidence_1 < box_confidence_2:
            new_y = new_y2
            box_confidence = box_confidence_2
            
        else:
            new_y = new_y1
            box_confidence = box_confidence_1
            
    else:
        new_y = new_y1
        box_confidence = box_confidence_1

    return new_x, new_y, box_confidence

# Main function to extract text with dynamic box adjustment
def extract_text_with_accuracy(image, Type):
    validated_texts = {}
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    total_confidence = 0
    valid_boxes = 0
    registration_number = ''
    
    # Iterate over each bounding box
    for i, (x, y, w, h) in enumerate(boxes):
        adjusted = False  # Keep track if the box was adjusted
        
        # Draw the bounding box for visualization
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the region of interest (ROI) from the image
        roi = image_cv[y:y + h, x:x + w]

        # Clear the image if Type is 1
        if Type == 2:
            roi = clear_image(roi)

        # Extract text using Tesseract OCR with confidence
        custom_config = r'--oem 3 --psm 6 -l eng'
        d = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract valid confidence values (ignoring -1)
        valid_confidences = [int(conf) for conf in d['conf'] if int(conf) != -1]

        # Only consider boxes with valid confidence values
        if valid_confidences:
            box_confidence = np.mean(valid_confidences)
            total_confidence += box_confidence
            valid_boxes += 1

            # Adjust the box if the confidence is too low
            if box_confidence < 30:
                xn, yn, box_confidence_n = adjust_box(x, y, w, h, image_cv, confidence_threshold=50)
                adjusted = True

        # Re-extract text after adjusting the box (if necessary)
        if adjusted:
            if(box_confidence_n > box_confidence):
                x, y, box_confidence = xn, yn, box_confidence_n
                    
            roi = image_cv[y:y + h, x:x + w]
            d = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)

        # Initialize detected_text with an empty string
        detected_text = ""

        # Extract text
        text = ' '.join([d['text'][i] for i in range(len(d['text'])) if int(d['conf'][i]) != -1])

        field_label = labels[i]

        # Add to detected texts
        if text:
            if field_label in word_lists:
                corrected_text = correct_word_ignore_case(text, word_lists[field_label])
                if corrected_text:
                    detected_text = corrected_text
                    total_confidence += 10
                else:
                    detected_text = text
            else:
                detected_text = text
        else:
            # If no text is extracted, default to empty string
            detected_text = ''

        # Validate the extracted text based on the field label
        validated_text = validate_extracted_text(field_label, detected_text, registration_number)

        # If the field is 'Registration No', update the registration_number variable
        if field_label == 'Registration No':
            registration_number = validated_text

        validated_texts[field_label] = validated_text

        if validated_texts[field_label] != text:
            total_confidence += 10

    # Calculate the overall accuracy based on valid boxes
    accuracy = total_confidence / valid_boxes if valid_boxes > 0 else 0

    return validated_texts, accuracy


if __name__ == '__main__':
    app.run(debug=True)
