from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from datetime import datetime


def generate_report(
    prediction,
    confidence,
    risk_level,
    metadata_found,
    explanation
):
    import os
    os.makedirs("outputs/reports", exist_ok=True)

    pdf_path = "outputs/reports/report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "TruthLens AI Forensic Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            f"<b>Prediction:</b> {prediction}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence}%",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Risk Level:</b> {risk_level}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Metadata Available:</b> {metadata_found}",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            "<b>Forensic Explanation</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            explanation,
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            f"Generated On: {datetime.now()}",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return pdf_path


if __name__ == "__main__":

    report = generate_report(
        prediction="deepfake",
        confidence=99.99,
        risk_level="HIGH",
        metadata_found=False,
        explanation="Example forensic explanation."
    )

    print("Report Saved:", report)