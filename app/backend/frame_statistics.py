from collections import Counter


def generate_frame_stats(predictions):

    counts = Counter(predictions)

    stats = {
        "Total Frames": len(predictions),
        "Deepfake Frames": counts.get("deepfake", 0),
        "Manipulated Frames": counts.get("manipulated", 0),
        "Real Frames": counts.get("real", 0)
    }

    return stats