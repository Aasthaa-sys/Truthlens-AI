import os
import matplotlib.pyplot as plt


def create_timeline(predictions):

    mapping = {
        "real": 0,
        "manipulated": 1,
        "deepfake": 2
    }

    values = [
        mapping[p]
        for p in predictions
    ]

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    save_path = "outputs/charts/timeline.png"

    plt.figure(figsize=(10, 4))

    plt.plot(
        values,
        marker="o"
    )

    plt.yticks(
        [0, 1, 2],
        ["Real", "Manipulated", "Deepfake"]
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Prediction"
    )

    plt.title(
        "Detection Timeline"
    )

    plt.grid(True)

    plt.savefig(
        save_path
    )

    plt.close()

    return save_path