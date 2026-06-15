def generate_explanation(
    prediction,
    confidence,
    metadata_found,
    risk_level
):

    explanation = []

    # ==========================
    # PREDICTION
    # ==========================

    explanation.append(
        f"The image was classified as {prediction.upper()} "
        f"with {confidence:.2f}% confidence."
    )

    # ==========================
    # CLASS SPECIFIC
    # ==========================

    if prediction == "deepfake":

        explanation.append(
            "The detection model found strong indicators "
            "of AI-generated or synthetic facial content."
        )

    elif prediction == "manipulated":

        explanation.append(
            "The image contains signs of digital editing "
            "or post-processing modifications."
        )

    else:

        explanation.append(
            "The image appears consistent with authentic "
            "camera-generated content."
        )

    # ==========================
    # METADATA
    # ==========================

    if metadata_found:

        explanation.append(
            "Metadata was detected in the image file, "
            "providing additional forensic information."
        )

    else:

        explanation.append(
            "No metadata was found in the image. "
            "Missing metadata can occur when images are "
            "edited, exported, compressed, or AI-generated."
        )

    # ==========================
    # ELA
    # ==========================

    explanation.append(
        "Error Level Analysis (ELA) was performed to "
        "identify unusual compression patterns that may "
        "indicate manipulation."
    )

    # ==========================
    # GRADCAM
    # ==========================

    explanation.append(
        "Grad-CAM visualization highlights the regions "
        "that most influenced the neural network's decision."
    )

    # ==========================
    # RISK
    # ==========================

    explanation.append(
        f"Overall forensic risk assessment: {risk_level}."
    )

    # ==========================
    # RETURN
    # ==========================

    return "\n\n".join(explanation)


if __name__ == "__main__":

    result = generate_explanation(
        prediction="deepfake",
        confidence=99.97,
        metadata_found=False,
        risk_level="HIGH"
    )

    print(result)