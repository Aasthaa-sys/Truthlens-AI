import os
import matplotlib.pyplot as plt


def create_confidence_graph(confidences):

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )
    os.makedirs("outputs/graphs", exist_ok=True)

    save_path = "outputs/charts/confidence_graph.png"

    plt.figure(figsize=(8, 4))

    plt.plot(
        confidences,
        marker="o"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Confidence (%)"
    )

    plt.title(
        "Confidence Across Frames"
    )

    plt.grid(True)

    plt.savefig(
        save_path
    )

    plt.close()

    return save_path