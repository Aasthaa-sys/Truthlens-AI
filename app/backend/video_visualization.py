import os
import matplotlib.pyplot as plt
from collections import Counter


def create_prediction_pie(predictions):

    counts = Counter(predictions)

    labels = list(counts.keys())
    sizes = list(counts.values())

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    save_path = "outputs/charts/prediction_pie.png"

    plt.figure(figsize=(6, 6))

    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title(
        "Frame Prediction Distribution"
    )

    plt.savefig(save_path)

    plt.close()

    return save_path