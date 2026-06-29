import cv2
import pytesseract
import shutil
import os
import platform

# ==========================
# TESSERACT PATH
# ==========================

if platform.system() == "Windows":
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
else:
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def detect_watermark(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return (False, [])

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    text = pytesseract.image_to_string(gray)
    text = text.lower()

    watermark_keywords = [
        "shutterstock",
        "getty",
        "adobe",
        "istock",
        "watermark",
        "dreamstime",
        "123rf"
    ]

    detected = []

    for keyword in watermark_keywords:
        if keyword in text:
            detected.append(keyword)

    if len(detected) > 0:
        return (True, detected)

    return (False, [])