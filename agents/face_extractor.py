import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


def extract_face(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detector.process(rgb)

    if not results.detections:
        return None

    detection = results.detections[0]

    bbox = detection.location_data.relative_bounding_box

    h, w, _ = frame.shape

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    padding = 20

    x = max(0, x - padding)
    y = max(0, y - padding)

    bw = bw + (2 * padding)
    bh = bh + (2 * padding)

    face = frame[y:y+bh, x:x+bw]

    return face