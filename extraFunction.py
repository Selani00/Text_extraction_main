from langdetect import LangDetectException
import cv2
import numpy as np


# Helper function to extract text from predefined boxes
def extract_text_from_boxes(image, Type):
    detected_texts = {}
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i, (x, y, w, h) in enumerate(boxes):
        # Draw the bounding box for visualization
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the region of interest (ROI) from the image
        roi = image_cv[y:y + h, x:x + w]

        # Clear the image if Type is 1
        if Type == 1:
            roi = clear_image(roi)

        # Extract text using Tesseract OCR
        custom_config = r'--oem 3 --psm 6 -l eng'
        text = pytesseract.image_to_string(roi, config=custom_config)

        # Clean and verify the extracted text using the classifier
        if text.strip():
            try:
                result = classifier(text, candidate_labels=[labels[i]])
                field_type = result['labels'][0]
                detected_texts[field_type] = text.strip()
            except LangDetectException:
                continue

    return detected_texts




# Function to deskew the image based on the skew angle
def deskew_image(image):
    # Convert the ROI to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use binary thresholding to highlight the text
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find the coordinates of non-zero pixels (i.e., the text)
    coords = np.column_stack(np.where(binary > 0))

    # Calculate the angle of the minimum area bounding box around the text
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed