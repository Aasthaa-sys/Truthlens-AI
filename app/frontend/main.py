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
from agents.llm_agent import generate_llm_explanation
from agents.deepfake_agent import predict_image
from agents.metadata_agent import extract_metadata
from agents.forensics_agent import perform_ela
from agents.risk_agent import assess_risk
from agents.explanation_agent import generate_explanation
from agents.report_agent import generate_report
from agents.video_classifier import predict_video
from agents.watermark_agent import detect_watermark
import agents.ocr_agent as ocr

from app.backend.gradcam import generate_gradcam
from app.backend.video_visualization import create_prediction_pie
from app.backend.confidence_graph import create_confidence_graph
from app.backend.frame_statistics import generate_frame_stats
from app.backend.detection_timeline import create_timeline

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="TruthLens AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3, h4 {
    color: #e2e8f0;
}

h1 {
    text-align: center;
    font-size: 2.4rem;
    letter-spacing: 1px;
}

p, li, label {
    color: #cbd5e1;
}

[data-testid="stMetricValue"] {
    font-size: 30px;
    color: #38bdf8;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 14px;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    height: 46px;
    width: 100%;
    font-size: 16px;
    border: none;
}

.stDownloadButton > button {
    background: #16a34a;
    color: white;
    border-radius: 8px;
    font-size: 15px;
    border: none;
}

[data-testid="stSidebar"] {
    background-color: #1e293b;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #e2e8f0;
}

div[data-testid="stTabs"] button {
    font-size: 15px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.markdown("## TruthLens AI")
st.sidebar.caption("AI-Powered Digital Forensics Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "",
    [
        "Dashboard",
        "Image Analysis",
        "Video Analysis",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Technologies")
st.sidebar.markdown("""
- PyTorch + ResNet50
- Grad-CAM (Explainable AI)
- Error Level Analysis
- OCR (Tesseract)
- Watermark Detection
- Streamlit + OpenCV
""")

st.sidebar.markdown("---")
st.sidebar.caption("TruthLens AI v2.0")

# ==========================
# HERO BANNER
# ==========================

st.markdown("""
<h1>TruthLens AI</h1>
<p style='text-align:center; color:#94a3b8; font-size:17px;'>
Explainable Multi-Agent Deepfake Detection and Digital Forensics Platform
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# TOP METRICS
# ==========================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model Accuracy", "96.55%")

with c2:
    st.metric("AI Agents", "8")

with c3:
    st.metric("Classes", "3")

with c4:
    st.metric("Video Support", "Yes")

st.markdown("---")

# ==========================
# DASHBOARD PAGE
# ==========================

if page == "Dashboard":

    st.markdown("### Processing Pipeline")

    st.info(
        "Upload -> Deepfake Detection -> Metadata Analysis -> "
        "Watermark Detection -> OCR -> ELA -> Grad-CAM -> "
        "Risk Assessment -> Report Generation"
    )

    st.markdown("---")

    st.markdown("### What TruthLens AI Detects")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.success("Deepfake Images")
        st.success("AI Generated Content")
        st.success("Manipulated Media")

    with d2:
        st.warning("Hidden Watermarks")
        st.warning("Embedded Text via OCR")
        st.warning("Metadata Anomalies")

    with d3:
        st.error("High Risk Content")
        st.error("Tampered Images")
        st.error("Forensic Irregularities")

    st.stop()

# ==========================
# ABOUT PAGE
# ==========================

if page == "About":

    st.markdown("### About TruthLens AI")

    st.markdown("""
    TruthLens AI is a Multi-Agent Explainable Deepfake Detection Framework
    built for digital forensics and media authenticity verification.

    **Architecture**

    The system runs a sequence of specialized AI agents on every uploaded file:
    """)

    st.markdown("""
    1. **Deepfake Agent** - ResNet50-based classifier trained on real, deepfake and AI-generated images
    2. **Metadata Agent** - Extracts and analyzes EXIF and file metadata
    3. **Watermark Agent** - Detects visible and hidden watermarks
    4. **OCR Agent** - Extracts embedded text using Tesseract
    5. **ELA Agent** - Error Level Analysis to detect compression tampering
    6. **Grad-CAM** - Produces visual heatmaps showing which regions influenced the prediction
    7. **Risk Agent** - Aggregates signals to assign a risk level
    8. **Report Agent** - Generates a downloadable PDF forensic report
    """)

    st.markdown("---")

    st.markdown("**Built using:** PyTorch, ResNet50, OpenCV, Streamlit, Tesseract, Grad-CAM")

    st.stop()

# ==========================
# FILE UPLOAD
# ==========================

st.markdown("### Upload Evidence")

uploaded_file = st.file_uploader(
    "Accepted formats: JPG, PNG, MP4",
    type=["jpg", "jpeg", "png", "mp4"]
)

# ==========================
# ANALYSIS
# ==========================

if uploaded_file:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    # ======================
    # VIDEO ANALYSIS
    # ======================

    if file_extension == "mp4":

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        progress = st.progress(0)
        st.caption("Running video analysis agents...")

        progress.progress(20)

        prediction, confidence, predictions, confidences = predict_video(temp_path)
        confidence = float(confidence)

        progress.progress(60)

        pie_chart = create_prediction_pie(predictions)
        confidence_graph = create_confidence_graph(confidences)
        timeline = create_timeline(predictions)
        stats = generate_frame_stats(predictions)

        progress.progress(100)

        st.success("Video analysis complete.")
        st.markdown("---")

        st.markdown("### Video Detection Results")

        v1, v2, v3 = st.columns(3)

        with v1:
            st.metric("Prediction", str(prediction).upper())

        with v2:
            st.metric("Confidence", f"{confidence:.2f}%")

        with v3:
            st.metric("Frames Analysed", len(predictions))

        st.progress(min(max(confidence / 100, 0), 1))
        st.caption(f"Confidence Score: {confidence:.2f}%")

        st.markdown("---")

        st.markdown("### Frame Statistics")
        st.write(stats)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Prediction Distribution")
            st.image(pie_chart)

        with col2:
            st.markdown("#### Detection Timeline")
            st.image(timeline)

        st.markdown("#### Confidence Graph")
        st.image(confidence_graph)

        st.markdown("---")

        st.markdown("""
        <p style='text-align:center; color:#64748b; font-size:13px;'>
        TruthLens AI - Powered by ResNet50 + Multi-Agent Framework
        </p>
        """, unsafe_allow_html=True)

        st.stop()

    # ======================
    # IMAGE ANALYSIS
    # ======================

    image_suffix = "." + file_extension

    with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    progress = st.progress(0)
    st.caption("Running AI agents...")

    progress.progress(10)
    prediction, confidence = predict_image(temp_path)
    confidence = float(confidence)

    progress.progress(25)
    metadata = extract_metadata(temp_path)
    metadata_found = len(metadata) > 0

    progress.progress(40)
    watermark_found, watermark_text = detect_watermark(temp_path)

    progress.progress(55)
    extracted_text = ocr.extract_text(temp_path)

    progress.progress(65)
    ela_path = perform_ela(temp_path)

    progress.progress(75)
    gradcam_path = generate_gradcam(temp_path)

    progress.progress(85)
    risk_level, reasons = assess_risk(
        prediction,
        confidence,
        metadata_found
    )

    if watermark_found:
        reasons.append("Watermark detected in image")

    progress.progress(92)
    explanation = generate_llm_explanation(

        prediction,

        confidence,

        risk_level,

        metadata_found,

        watermark_found,

        extracted_text

    )

    progress.progress(98)
    pdf_path = generate_report(
        prediction,
        confidence,
        risk_level,
        metadata_found,
        explanation
    )

    progress.progress(100)

    st.success("Analysis complete.")
    st.markdown("---")

    # ======================
    # TABS
    # ======================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Results",
        "Forensics",
        "Metadata",
        "Report"
    ])

    # ======================
    # TAB 1 - RESULTS
    # ======================

    with tab1:

        st.markdown("### Detection Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Prediction", str(prediction).upper())

        with r2:
            st.metric("Confidence", f"{confidence:.2f}%")

        with r3:
            st.metric("Risk Level", risk_level)

        st.progress(min(max(confidence / 100, 0), 1))
        st.caption(f"Confidence Score: {confidence:.2f}%")

        st.markdown("---")

        if str(prediction).lower() == "deepfake":
            st.error("Deepfake Detected - This image shows signs of AI manipulation.")
        elif str(prediction).lower() == "ai generated":
            st.warning("AI Generated Content - This image appears to be synthetically created.")
        else:
            st.success("Real Image - No deepfake indicators found.")

        st.markdown("---")

        st.markdown("### Risk Assessment")

        if risk_level == "HIGH":
            st.error(f"Risk Level: {risk_level}")
        elif risk_level == "MEDIUM":
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.success(f"Risk Level: {risk_level}")

        st.markdown("**Risk Factors:**")

        for reason in reasons:
            st.write(f"- {reason}")

        st.markdown("---")

        with st.expander("Forensic Explanation"):
            st.write(explanation)

    # ======================
    # TAB 2 - FORENSICS
    # ======================

    with tab2:

        st.markdown("### Image Forensics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original Image**")

            if os.path.exists(temp_path):
                st.image(
                    temp_path,
                    caption="Original",
                    use_container_width=True
                )
            else:
                st.warning("Original image not found.")

        with col2:
            st.markdown("**Error Level Analysis**")

            if os.path.exists(ela_path):
                st.image(
                    ela_path,
                    caption="ELA",
                    use_container_width=True
                )
            else:
                st.warning("ELA image could not be generated.")

        with col3:
            st.markdown("**Grad-CAM Heatmap**")

            if os.path.exists(gradcam_path):
                st.image(
                    gradcam_path,
                    caption="Grad-CAM",
                    use_container_width=True
                )
            else:
                st.warning("Grad-CAM image could not be generated.")

        st.markdown("---")

        st.markdown("### Watermark Detection")

        if watermark_found:
            st.error("Watermark Detected")
            st.write(watermark_text)
        else:
            st.success("No Watermark Found")

        st.markdown("---")

        with st.expander("OCR - Extracted Text"):

            if extracted_text.strip():
                st.success("Text found in image.")
                st.text(extracted_text)
            else:
                st.info("No readable text detected in this image.")

    # ======================
    # TAB 3 - METADATA
    # ======================

    with tab3:

        st.markdown("### Metadata Analysis")

        if metadata_found:

            st.success("Metadata found.")

            with st.expander("View Metadata"):

                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")

        else:
            st.warning("No metadata found. This may indicate the file has been stripped or re-encoded.")

    # ======================
    # TAB 4 - REPORT
    # ======================

    with tab4:

        st.markdown("### Forensic Report")

        st.write(
            "The report below contains the full analysis including prediction, "
            "confidence score, risk level, metadata status and AI-generated explanation."
        )

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Forensic Report (PDF)",
                    data=pdf_file,
                    file_name="TruthLens_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Report could not be generated.")

    st.markdown("---")

    st.markdown("""
    <p style='text-align:center; color:#64748b; font-size:13px;'>
    TruthLens AI - Powered by ResNet50, Grad-CAM, OpenCV, PyTorch and Streamlit
    </p>
    """, unsafe_allow_html=True)