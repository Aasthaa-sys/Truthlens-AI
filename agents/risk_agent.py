def assess_risk(
    prediction,
    confidence,
    metadata_found
):

    risk_level = "LOW"

    reasons = []

    # Deepfake detection
    if prediction == "deepfake":

        risk_level = "HIGH"

        reasons.append(
            "Deepfake content detected"
        )

    # Manipulated image
    elif prediction == "manipulated":

        risk_level = "MEDIUM"

        reasons.append(
            "Image manipulation detected"
        )

    # Confidence check
    if confidence > 95:

        reasons.append(
            "High confidence prediction"
        )

    # Metadata check
    if not metadata_found:

        reasons.append(
            "Metadata unavailable"
        )

    return risk_level, reasons


if __name__ == "__main__":

    prediction = "deepfake"

    confidence = 99.97

    metadata_found = False

    risk, reasons = assess_risk(
        prediction,
        confidence,
        metadata_found
    )

    print("\n===== FORENSIC REPORT =====")

    print("Prediction:", prediction)

    print("Confidence:", confidence)

    print("Risk Level:", risk)

    print("\nReasons:")

    for reason in reasons:

        print("-", reason)