
import cv2
import os


def extract_frames(
    video_path,
    output_folder="outputs/video_frames"
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    count = 0
    saved = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Save 1 frame every 30 frames
        if count % 30 == 0:

            frame_path = os.path.join(
                output_folder,
                f"frame_{saved}.jpg"
            )

            cv2.imwrite(
                frame_path,
                frame
            )

            saved += 1

        count += 1

    cap.release()

    print(
        f"{saved} frames extracted"
    )

    return output_folder


if __name__ == "__main__":

    video_path = r"C:\Users\study\Downloads\sample_video (360p).mp4"

    folder = extract_frames(
        video_path
    )

    print(
        "\nFrames Saved To:"
    )

    print(folder)