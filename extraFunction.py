import cv2
import numpy as np


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