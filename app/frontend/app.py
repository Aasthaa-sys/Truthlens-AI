import streamlit as st
import tempfile
import os
import sys

# ==========================
# PROJECT PATH
# ==========================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

# ==========================
# IMPORT AGENTS
# ==========================
import os
import sys

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(project_root)

from agents.deepfake_agent import predict_image
from agents.risk_agent import assess_risk
from agents.metadata_agent import extract_metadata
from agents.forensics_agent import perform_ela
from backend.gradcam import generate_gradcam

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="TruthLens AI",
    layout="wide"
)

# ==========================
# TITLE
# ==========================

st.title("TruthLens AI")

st.subheader(
    "Multi-Agent Deepfake Detection System"
)

st.markdown("---")

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# ANALYSIS
# ==========================

if uploaded_file:

    # ======================
    # SAVE TEMP IMAGE
    # ======================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_path = temp_file.name

    # ======================
    # DEEPFAKE AGENT
    # ======================

    prediction, confidence = predict_image(
        temp_path
    )

    # ======================
    # METADATA AGENT
    # ======================

    metadata = extract_metadata(
        temp_path
    )

    metadata_found = (
        len(metadata) > 0
    )

    # ======================
    # ELA AGENT
    # ======================

    ela_path = perform_ela(
        temp_path
    )

    # ======================
    # GRADCAM AGENT
    # ======================

    gradcam_path = generate_gradcam(
        temp_path
    )

    # ======================
    # RISK AGENT
    # ======================

    risk_level, reasons = assess_risk(
        prediction,
        confidence,
        metadata_found
    )

    # ======================
    # FORENSIC VISUALS
    # ======================

    st.subheader(
        "Image Forensics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            temp_path,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            ela_path,
            caption="ELA Analysis",
            use_container_width=True
        )

    with col3:

        st.image(
            gradcam_path,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    st.markdown("---")

    # ======================
    # RESULTS
    # ======================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.subheader(
            "Detection Results"
        )

        st.metric(
            "Prediction",
            prediction.upper()
        )

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

    with result_col2:

        st.subheader(
            "Risk Assessment"
        )

        if risk_level == "HIGH":

            st.error(
                f"Risk Level: {risk_level}"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            st.success(
                f"Risk Level: {risk_level}"
            )

        st.write("Reasons:")

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

    st.markdown("---")

    # ======================
    # METADATA
    # ======================

    st.subheader(
        "Metadata Analysis"
    )

    if metadata_found:

        st.success(
            "Metadata Found"
        )

        for key, value in metadata.items():

            st.write(
                f"**{key}** : {value}"
            )

    else:

        st.warning(
            "No Metadata Found"
        )

    st.markdown("---")

    st.info(
        "Powered by ResNet50 + Multi-Agent AI Framework"
    )