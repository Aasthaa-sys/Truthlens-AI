import cv2
import pytesseract
import os
import platform

# ==========================
# TESSERACT CONFIGURATION
# ==========================

# Windows (Local Development)
if platform.system() == "Windows":

    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Linux (Streamlit Cloud / Render)
# No path required if Tesseract is installed using packages.txt

# ==========================
# OCR FUNCTION
# ==========================

def extract_text(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Improve text visibility
    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # OCR
    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    return text.strip()