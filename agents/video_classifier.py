import os
import cv2

from agents.face_extractor import extract_face
from agents.frame_preprocessor import enhance_frame
from agents.video_agent import extract_frames
from agents.deepfake_agent import predict_image

os.makedirs("outputs/videos", exist_ok=True)

def predict_video(video_path):

    frames_folder = extract_frames(
        video_path
    )

    predictions = []
    confidences = []

    for frame_name in sorted(
        os.listdir(frames_folder)
    ):

        frame_path = os.path.join(
            frames_folder,
            frame_name
        )

        frame = cv2.imread(
            frame_path
        )

        if frame is None:
            continue

        face = extract_face(
            frame
        )

        if face is None:
            continue

        face = enhance_frame(
            face
        )
        
        if face.shape[0] < 50 or face.shape[1] < 50:
            continue


        temp_face = "temp_face.jpg"

        cv2.imwrite(
            temp_face,
            face
        )

        
        prediction, confidence = predict_image(
            temp_face
        )

        if confidence < 70:
            continue

        predictions.append(
            prediction
        )

        confidences.append(
            confidence
        )

        print(
            frame_name,
            "->",
            prediction,
            f"({confidence}%)"
        )

    if len(predictions) == 0:

        return (
            "unknown",
            0,
            [],
            []
        )

    deepfake_score = 0
    manipulated_score = 0
    real_score = 0

    for pred, conf in zip(
        predictions,
        confidences
    ):

        if pred == "deepfake":

            deepfake_score += conf

        elif pred == "manipulated":

            manipulated_score += conf

        else:

            real_score += conf

    scores = {
        "deepfake": deepfake_score,
        "manipulated": manipulated_score,
        "real": real_score
    }

    final_prediction = max(
        scores,
        key=scores.get
    )

    total_score = sum(
        scores.values()
    )

    if total_score == 0:

        final_confidence = 0

    else:

        final_confidence = round(
            (
                scores[final_prediction]
                /
                total_score
            ) * 100,
            2
        )

    return (
        final_prediction,
        final_confidence,
        predictions,
        confidences
    )


if __name__ == "__main__":

    video_path = r"C:\Users\study\Downloads\61706-500316063.mp4"

    (
        prediction,
        confidence,
        predictions,
        confidences
    ) = predict_video(
        video_path
    )

    print("\n===================")

    print(
        "VIDEO RESULT:"
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence,
        "%"
    )

    print(
        "Total Frames:",
        len(predictions)
    )