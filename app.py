import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="Nail Disease Segmentation", layout="wide", initial_sidebar_state="collapsed")

# --- Configuration ---
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.20
MASK_ALPHA = 0.5
PROJECT_GROUP_NAME = "youngstunna"

# --- Inject Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset and base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    height: 100%;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Full-page app background */
[data-testid="stAppViewContainer"] {
    background: #f0f2f6;
    padding: 0 !important;
}

[data-testid="stAppViewBody"] {
    padding: 0 !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
}

/* === TWO-COLUMN LAYOUT === */
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    align-items: stretch !important;
    min-height: 100vh;
}

/* LEFT COLUMN — Blue */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    background: #1a3a8f !important;
    padding: 0 !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > div {
    background: #1a3a8f !important;
    padding: 3rem 2.5rem !important;
    min-height: 100vh;
}

/* Force all text in left column to white */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child * {
    color: white !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child h1,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child h2,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child h3 {
    color: white !important;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.2;
    margin-bottom: 1.5rem;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child p,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child li {
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.95rem;
    line-height: 1.7;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child hr {
    border-color: rgba(255,255,255,0.25) !important;
    margin: 1.5rem 0;
}

/* LEFT column step numbers */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1rem;
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.95rem;
}
.step-num {
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    width: 28px;
    height: 28px;
    min-width: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.8rem;
    color: white !important;
}
.disclaimer-text {
    font-style: italic;
    font-size: 0.82rem !important;
    color: rgba(255,255,255,0.6) !important;
    margin-top: 1rem;
    line-height: 1.5;
}
.section-label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.5) !important;
    margin-bottom: 1rem !important;
    margin-top: 0.5rem !important;
}
.logo-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

/* RIGHT COLUMN — White */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    background: #ffffff !important;
    padding: 0 !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > div {
    background: #ffffff !important;
    padding: 3rem 2.5rem !important;
    min-height: 100vh;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child h2,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child h3 {
    color: #1a1a2e !important;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child p,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child label {
    color: #444 !important;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    border: 2px dashed #c5cfe8 !important;
    border-radius: 10px !important;
    background: #f7f9ff !important;
    padding: 1.5rem !important;
    transition: all 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #1a3a8f !important;
    background: #eef1fb !important;
}

[data-testid="stFileUploader"] label {
    color: #333 !important;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"] {
    background: #1a3a8f !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #142d72 !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    color: #1a3a8f !important;
}

/* Alerts */
[data-testid="stSuccess"] { border-left: 4px solid #1a3a8f !important; }
[data-testid="stWarning"] { border-left: 4px solid #f0a500 !important; }
[data-testid="stInfo"] { border-left: 4px solid #5b8dee !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
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


# === MAIN LAYOUT ===
col1, col2 = st.columns([1, 1.3], gap="small")

# ── LEFT COLUMN ──
with col1:
    st.markdown('<div class="logo-icon">🔬</div>', unsafe_allow_html=True)
    st.markdown("## Nail Disease Segmentation")

    st.write(
        f"An AI-powered application developed by **Group {PROJECT_GROUP_NAME}** for the AI2 T1 AY2526 course. "
        "This tool analyzes nail images to identify potential health conditions."
    )

    st.markdown("---")

    st.markdown('<p class="section-label">How it works</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-item">
        <div class="step-num">1</div>
        <span>Upload a clear image of a nail.</span>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <span>The AI model analyzes the image.</span>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <span>Detected conditions (or healthy status) are highlighted.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="disclaimer-text">Disclaimer: This tool is for educational purposes only and not a substitute for professional medical diagnosis.</p>',
        unsafe_allow_html=True
    )


# ── RIGHT COLUMN ──
with col2:
    st.markdown("### Analyze Your Image")

    model = load_yolo_model(MODEL_PATH)

    if model is None:
        st.error(
            f"FATAL ERROR: Model failed to load from path '{MODEL_PATH}'. "
            "Ensure 'best.pt' is in the application directory and not corrupted."
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
                            overlay_h, overlay_w = overlay_image.shape[:2]

                            for i, mask in enumerate(masks):
                                if i < len(class_ids):
                                    mask_resized = cv2.resize(mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
                                    mask_uint8 = mask_resized.astype(np.uint8) * 255
                                    class_id = class_ids[i]
                                    if 0 <= class_id < len(colors):
                                        color = colors[class_id]
                                        colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
                                        for c_idx in range(3):
                                            colored_mask[:, :, c_idx] = np.where(mask_uint8 == 255, color[c_idx], 0)
                                        overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, MASK_ALPHA, 0)

                        if class_ids.size > 0:
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                if i < len(class_ids):
                                    cls_id = class_ids[i]
                                    if 0 <= cls_id < len(names):
                                        x1, y1, x2, y2 = map(int, box)
                                        label = f"{names[cls_id]} {score:.2f}"
                                        color = colors[cls_id]
                                        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
                                        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        y_bg_top = max(0, y1 - lh - baseline - 5)
                                        cv2.rectangle(overlay_image, (x1, y_bg_top), (x1 + lw, y1), color, cv2.FILLED)
                                        cv2.putText(overlay_image, label, (x1, max(10, y1 - baseline - 3)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                                    lineType=cv2.LINE_AA)

                if detection_made:
                    is_only_healthy = detected_classes == {'healthy_nail'}
                    result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

                    if is_only_healthy:
                        message_placeholder.success("✅ Healthy nail detected. No diseases found.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image — Healthy Nail', use_container_width=True)
                    else:
                        message_placeholder.warning("⚠️ Nail condition(s) detected. This is not a medical diagnosis. Please consult a healthcare professional.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image — Detections Highlighted', use_container_width=True)
                else:
                    message_placeholder.info(
                        f"No nail conditions were detected above the {CONFIDENCE_THRESHOLD * 100:.0f}% confidence threshold."
                    )
                    result_placeholder.image(image, caption='Uploaded Image', use_container_width=True)

            except Exception as e:
                result_placeholder.empty()
                message_placeholder.empty()
                st.error(f"An error occurred during image processing: {e}")
                st.warning("Please ensure you uploaded a valid image file (JPG, PNG, JPEG).")


# --- Sidebar ---
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
st.sidebar.info("Users are expected to follow responsible research and innovation practices, ensuring that any derivative work is:")
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
