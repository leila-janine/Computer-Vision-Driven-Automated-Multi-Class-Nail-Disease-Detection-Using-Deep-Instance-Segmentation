import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Nail Disease Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
CONFIDENCE_THRESHOLD = 0.20
MASK_ALPHA = 0.5
PROJECT_GROUP_NAME = "youngstunna"

# --- Inject Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── DARK APP BACKGROUND ── */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main {
    background-color: #0e1117 !important;
}

.main .block-container {
    background-color: #0e1117 !important;
    padding: 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #1a1d27 !important;
    border-right: 1px solid #2d3748 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f7fafc !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #a0aec0 !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #2d3748 !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #1e2535 !important;
    border: 1px solid #2d3748 !important;
}

/* ── MAIN CONTENT TEXT ── */
.main h1, .main h2, .main h3 {
    color: #f7fafc !important;
    font-weight: 700 !important;
}

.main p, .main div, .main li, .main label, .main span {
    color: #cbd5e0 !important;
}

/* ── DIVIDER ── */
.main hr {
    border-color: #2d3748 !important;
    margin: 1.5rem 0 !important;
}

/* ── CUSTOM HTML CLASSES ── */
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f7fafc !important;
    margin-bottom: 0.75rem;
    line-height: 1.2;
}

.main-subtitle {
    color: #a0aec0 !important;
    font-size: 0.93rem;
    line-height: 1.75;
    margin-bottom: 1.5rem;
}

.how-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #718096 !important;
    margin-bottom: 0.6rem;
}

.step-row {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    margin-bottom: 0.5rem;
    color: #a0aec0;
    font-size: 0.9rem;
}

.step-dot {
    color: #718096;
    font-size: 0.85rem;
    min-width: 20px;
    padding-top: 1px;
}

.disclaimer {
    font-size: 0.8rem;
    color: #718096 !important;
    font-style: italic;
    margin-top: 1.25rem;
    line-height: 1.5;
}

.right-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f7fafc !important;
    margin-bottom: 1.25rem;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background-color: #1a1d27 !important;
    border: 1.5px dashed #3a4a6b !important;
    border-radius: 8px !important;
    padding: 1.1rem 1.25rem !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #5b7fcf !important;
}

[data-testid="stFileUploader"] label {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Browse files button */
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] button {
    background-color: #2d3748 !important;
    color: #e2e8f0 !important;
    border: 1px solid #4a5568 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1.1rem !important;
    transition: all 0.2s !important;
}

[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stFileUploader"] button:hover {
    background-color: #3d4f6e !important;
    border-color: #5b7fcf !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] > div {
    color: #5b7fcf !important;
}

/* ── ALERTS ── */
[data-testid="stSuccess"],
[data-testid="stSuccess"] * {
    background-color: #1a2e1a !important;
    border: 1px solid #276227 !important;
    color: #9ae6b4 !important;
}

[data-testid="stWarning"],
[data-testid="stWarning"] * {
    background-color: #2d2010 !important;
    border: 1px solid #744210 !important;
    color: #fbd38d !important;
}

[data-testid="stInfo"],
[data-testid="stInfo"] * {
    background-color: #172136 !important;
    border: 1px solid #2b4c7e !important;
    color: #90cdf4 !important;
}

[data-testid="stError"],
[data-testid="stError"] * {
    background-color: #2d1515 !important;
    border: 1px solid #742424 !important;
    color: #feb2b2 !important;
}

/* ── COLUMNS ── */
[data-testid="stHorizontalBlock"] {
    gap: 3rem !important;
    align-items: flex-start !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    border-right: 1px solid #2d3748;
    padding-right: 2rem !important;
}

/* ── IMAGES ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
st.sidebar.title("Ethical Considerations")
st.sidebar.markdown("---")

st.sidebar.subheader("Notice on Use, Redistribution, and Ethical Compliance")
st.sidebar.warning(
    "Redistribution, reproduction, or use of this material beyond personal reference is strictly prohibited "
    "without the prior written consent of the author. Unauthorized copying, modification, or dissemination—"
    "whether for commercial, academic, or institutional purposes—violates intellectual property rights and may "
    "result in legal or disciplinary action."
)

st.sidebar.subheader("AI Governance and Ethics Considerations")
st.sidebar.error("This work must not be used in ways that:")
st.sidebar.markdown("""
* Compromise data privacy or violate data protection regulations (e.g., GDPR, Philippine Data Privacy Act).
* Perpetuate bias or discrimination by misusing algorithms, datasets, or results.
* Enable harmful applications, including surveillance, profiling, or uses that undermine human rights.
* Misrepresent authorship or credit, such as plagiarism or omission of proper citations.
""")

st.sidebar.subheader("Responsible Use Principles")
st.sidebar.info(
    "Users are expected to follow responsible research and innovation practices, "
    "ensuring that any derivative work is:"
)
st.sidebar.markdown("""
* **Transparent** → Clear acknowledgment of sources and methodology.
* **Accountable** → Proper attribution and disclosure of limitations.
* **Beneficial to society** → Applications that align with ethical standards and do not cause harm.
""")
st.sidebar.markdown("---")
st.sidebar.caption(
    "For any intended use (academic, research, or practical), prior written approval must be obtained "
    "from the author to ensure compliance with both legal requirements and ethical AI practices."
)


# ══════════════════════════════════════════
#  MAIN CONTENT — TWO COLUMNS
# ══════════════════════════════════════════
col1, col2 = st.columns([1, 1.1], gap="large")

# ── LEFT COLUMN ──
with col1:
    st.markdown(f'<p class="main-title">Nail Disease Segmentation</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="main-subtitle">'
        f'An AI-powered application developed by <strong style="color:#e2e8f0">Group {PROJECT_GROUP_NAME}</strong> '
        f'for the AI2 T1 AY2526 course. This tool analyzes nail images to identify potential health conditions. '
        f'Upload an image, and the system will attempt to segment and classify areas indicating specific nail '
        f'diseases or confirm healthy nails.'
        f'</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown('<p class="how-label">How it works:</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-row"><span class="step-dot">1.</span><span>Upload a clear image of a nail.</span></div>
    <div class="step-row"><span class="step-dot">2.</span><span>The AI model analyzes the image.</span></div>
    <div class="step-row"><span class="step-dot">3.</span><span>Detected conditions (or healthy status) are highlighted.</span></div>
    <p class="disclaimer">Disclaimer: This tool is for educational purposes only and not a substitute for professional medical diagnosis.</p>
    """, unsafe_allow_html=True)


# ── RIGHT COLUMN ──
with col2:
    st.markdown('<p class="right-title">Analyze Your Image</p>', unsafe_allow_html=True)

    model = load_yolo_model(MODEL_PATH)

    if model is None:
        st.error(
            f"FATAL ERROR: Model failed to load from '{MODEL_PATH}'. "
            "Ensure 'best.pt' is in the application directory."
        )
        st.stop()
    else:
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)...",
            type=["jpg", "png", "jpeg"],
            label_visibility="visible"
        )

        if uploaded_file is not None:
            result_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                img_cv = np.array(image.convert('RGB'))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                result_placeholder.image(image, caption='Uploaded Image', use_container_width=True)

                with st.spinner("Analyzing image..."):
                    results = model(img_cv, conf=CONFIDENCE_THRESHOLD)

                    overlay_image = img_cv.copy()
                    detection_made = False
                    detected_classes = set()

                    names = model.names
                    np.random.seed(42)
                    colors = [tuple(np.random.randint(80, 240, 3).tolist()) for _ in range(len(names))]

                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()

                        if boxes.shape[0] > 0:
                            class_ids = r.boxes.cls.cpu().numpy().astype(int)
                            detection_made = True
                            for cls_id in class_ids:
                                if 0 <= cls_id < len(names):
                                    detected_classes.add(names[cls_id])
                        else:
                            class_ids = np.array([], dtype=int)

                        if r.masks is not None and len(class_ids) > 0:
                            masks = r.masks.data.cpu().numpy()
                            oh, ow = overlay_image.shape[:2]
                            for i, mask in enumerate(masks):
                                if i < len(class_ids):
                                    m = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
                                    m8 = m.astype(np.uint8) * 255
                                    cid = class_ids[i]
                                    if 0 <= cid < len(colors):
                                        color = colors[cid]
                                        cm = np.zeros_like(overlay_image, dtype=np.uint8)
                                        for c in range(3):
                                            cm[:, :, c] = np.where(m8 == 255, color[c], 0)
                                        overlay_image = cv2.addWeighted(overlay_image, 1.0, cm, MASK_ALPHA, 0)

                        if class_ids.size > 0:
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                if i < len(class_ids):
                                    cid = class_ids[i]
                                    if 0 <= cid < len(names):
                                        x1, y1, x2, y2 = map(int, box)
                                        label = f"{names[cid]} {score:.2f}"
                                        color = colors[cid]
                                        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
                                        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        y_bg = max(0, y1 - lh - bl - 5)
                                        cv2.rectangle(overlay_image, (x1, y_bg), (x1 + lw, y1), color, cv2.FILLED)
                                        cv2.putText(overlay_image, label, (x1, max(10, y1 - bl - 3)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                                    lineType=cv2.LINE_AA)

                result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

                if detection_made:
                    if detected_classes == {'healthy_nail'}:
                        message_placeholder.success("✅ Healthy nail detected. No diseases found.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image — Healthy Nail', use_container_width=True)
                    else:
                        message_placeholder.warning("⚠️ Nail condition(s) detected. Not a medical diagnosis — consult a healthcare professional.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image — Detections Highlighted', use_container_width=True)
                else:
                    message_placeholder.info(
                        f"No conditions detected above the {CONFIDENCE_THRESHOLD * 100:.0f}% confidence threshold."
                    )
                    result_placeholder.image(image, caption='Uploaded Image', use_container_width=True)

            except Exception as e:
                result_placeholder.empty()
                message_placeholder.empty()
                st.error(f"An error occurred during processing: {e}")
                st.warning("Please ensure you uploaded a valid image file (JPG, PNG, JPEG).")
