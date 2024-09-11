import cv2
import numpy as np
import pandas as pd
import difflib
from datetime import datetime
import os

def correct_word_ignore_case(word, possibilities):
    word_lower = word.lower()
    possibilities_lower = [possible_word.lower() for possible_word in possibilities]
    matches = difflib.get_close_matches(word_lower, possibilities_lower, n=1, cutoff=0.6)
    
    if matches:
        # Find the original case-sensitive word from the possibilities list
        original_word = possibilities[possibilities_lower.index(matches[0])]
        return original_word
    return None

# Helper function to save data to the same Excel file
def save_data_to_excel(detected_texts, accuracy, visibility, image_id):
    # Define the Excel file name
    excel_filename = 'detected_text_data_n.xlsx'
    
    # Check if the file already exists
    if os.path.exists(excel_filename):
        # Load the existing file
        df_existing = pd.read_excel(excel_filename, engine='openpyxl')
        # Determine the next auto-increment number
        next_auto_increment = len(df_existing) + 1
    else:
        # If the file doesn't exist, start with auto-increment number 1
        next_auto_increment = 1
        df_existing = pd.DataFrame()

    # Prepare the new row data
    new_row_data = {
        'ID': next_auto_increment,
        'Date and Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Image_ID': image_id,
        'Accuracy': f"{accuracy:.2f}%",
        'Visibility': f"{visibility:.2f}%"
    }
    
    # Add the detected text fields to the new row
    for field, text in detected_texts.items():
        new_row_data[field] = text

    # Convert the new row data to a DataFrame
    df_new_row = pd.DataFrame([new_row_data])

    # Append the new row to the existing DataFrame
    df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)

    # Save the combined DataFrame back to the Excel file
    df_combined.to_excel(excel_filename, index=False, engine='openpyxl')

    print(f"Data appended to {excel_filename}")


# Helper function to calculate visibility percentage
def calculate_visibility(image):
    # Convert to OpenCV format for image processing
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Assuming that "visibility" is based on the proportion of non-white pixels
    total_pixels = gray.size
    non_white_pixels = cv2.countNonZero(gray)
    visibility_percentage = (non_white_pixels / total_pixels) * 100

    return visibility_percentage

# Helper function to clear the image
def clear_image(image):

    print("Clearing the image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    text_only = cv2.subtract(binary, detected_lines)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    text_without_vertical_lines = cv2.subtract(text_only, detected_vertical_lines)

    result = cv2.bitwise_not(text_without_vertical_lines)
    return result


# Helper function to identify the type of the image
def identify_image_type(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ref_image1 = cv2.imread('Images/1.jpg')
    ref_image2 = cv2.imread('Images/_1.jpg')

    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray_ref_image1 = cv2.cvtColor(ref_image1, cv2.COLOR_BGR2GRAY)
    gray_ref_image2 = cv2.cvtColor(ref_image2, cv2.COLOR_BGR2GRAY)

    edges_image = cv2.Canny(gray_image, 50, 150)
    edges_ref_image1 = cv2.Canny(gray_ref_image1, 50, 150)
    edges_ref_image2 = cv2.Canny(gray_ref_image2, 50, 150)

    num_edges_image = cv2.countNonZero(edges_image)
    num_edges_ref_image1 = cv2.countNonZero(edges_ref_image1)
    num_edges_ref_image2 = cv2.countNonZero(edges_ref_image2)

    if abs(num_edges_image - num_edges_ref_image1) < abs(num_edges_image - num_edges_ref_image2):
        return 1
    else:
        return 2