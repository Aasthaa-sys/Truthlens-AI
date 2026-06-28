import cv2
import pytesseract
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def detect_watermark(image_path):

    image = cv2.imread(
        image_path
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    text = pytesseract.image_to_string(
        gray
    )

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

            detected.append(
                keyword
            )

    if len(detected) > 0:

        return (
            True,
            detected
        )

    return (
        False,
        []
    )
