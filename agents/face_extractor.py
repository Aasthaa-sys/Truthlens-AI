import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def extract_face(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        4
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    padding = 20

    x = max(0, x - padding)
    y = max(0, y - padding)

    w = w + (2 * padding)
    h = h + (2 * padding)

    face = frame[y:y+h, x:x+w]

    return face