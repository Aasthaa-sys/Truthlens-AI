import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)
from agents.video_classifier import predict_video

from video_visualization import create_prediction_pie
from confidence_graph import create_confidence_graph
from frame_statistics import generate_frame_stats
from detection_timeline import create_timeline


def analyze_video(video_path):

    (
        final_prediction,
        final_confidence,
        predictions,
        confidences
    ) = predict_video(
        video_path
    )

    pie_chart = create_prediction_pie(
        predictions
    )

    confidence_graph = create_confidence_graph(
        confidences
    )

    timeline = create_timeline(
        predictions
    )

    stats = generate_frame_stats(
        predictions
    )

    return (
        final_prediction,
        final_confidence,
        stats,
        pie_chart,
        confidence_graph,
        timeline
    )


if __name__ == "__main__":

    video_path = r"C:\Users\study\Downloads\sample_video (360p).mp4"

    (
        prediction,
        confidence,
        stats,
        pie_chart,
        confidence_graph,
        timeline
    ) = analyze_video(
        video_path
    )

    print("\n===== VIDEO ANALYTICS =====")

    print("Prediction:", prediction)

    print("Confidence:", confidence)

    print("\nStatistics:")

    for key, value in stats.items():

        print(key, ":", value)

    print("\nPie Chart:", pie_chart)

    print("Confidence Graph:", confidence_graph)

    print("Timeline:", timeline)
