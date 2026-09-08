import html
import io
import os
import pathlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TF prints noisy one-time startup notices; predictions are unaffected

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img

IMG_SIZE = (224, 224)  # MobileNetV2's native input size
MAX_UPLOAD_BYTES = 1 * 1024 * 1024  # 1 MB
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
# Same order Keras used at training time (folders sort alphabetically):
# paper -> 0, rock -> 1, scissors -> 2
CLASSES = ["paper", "rock", "scissors"]
GESTURE_EMOJI = {"paper": "✋", "rock": "✊", "scissors": "✌️"}

st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("model.h5")


try:
    model = load_model()
except Exception:
    st.error("Couldn't load `model.h5`. Run `model.ipynb` top to bottom first — it trains and saves the model.")
    st.stop()


def pad_zoom_out(pil_img, pad_frac):
    """Surround the image with its own background color, zooming the content out."""
    arr = np.asarray(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    bg = np.median(arr.reshape(-1, 3), axis=0).astype(np.uint8)
    ph, pw = int(h * pad_frac), int(w * pad_frac)
    canvas = np.full((ph, pw, 3), bg, dtype=np.uint8)
    y0, x0 = (ph - h) // 2, (pw - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    return Image.fromarray(canvas)


def call_the_throw(pil_img):
    """Predict with a small padding ensemble (test-time augmentation).

    The model is sensitive to how much of the frame the hand fills, because
    the training photos all have the same framing. Predicting on the original
    plus mildly zoomed-out variants and averaging the probabilities makes the
    verdict stable across distances: close-up hands gain a little context,
    distant hands stop being dominated by the sudden scene change at the
    frame edge. Same preprocessing as training otherwise: RGB, 224x224,
    MobileNetV2 [-1, 1] scaling.
    """
    probs = np.zeros(3, dtype=np.float64)
    for pad_frac in (1.0, 1.15, 1.3):
        variant = pil_img if pad_frac == 1.0 else pad_zoom_out(pil_img, pad_frac)
        x = preprocess_input(
            np.asarray(variant.convert("RGB").resize(IMG_SIZE, Image.NEAREST), dtype=np.float32)
        )
        probs += model.predict(np.expand_dims(x, 0), verbose=0)[0]
    return probs / 3


def render_probs(probs) -> str:
    """Probability rows for the result card. Losers render dimmed."""
    winner = CLASSES[int(np.argmax(probs))]
    rows = ""
    for cls, p in zip(CLASSES, probs):
        dim = "" if cls == winner else "dim"
        emoji = GESTURE_EMOJI[cls]
        rows += f"""
        <div class="prob-row {dim}">
          <span class="prob-label"><span class="g-emoji">{emoji}</span>{cls}</span>
          <div class="prob-track"><div class="prob-fill f-{cls}" style="width:{p * 100:.1f}%"></div></div>
          <span class="prob-pct">{p * 100:.0f}%</span>
        </div>"""
    return rows


def render_result(probs) -> str:
    """The result card: called throw, confidence, per-class probabilities."""
    winner = CLASSES[int(np.argmax(probs))]
    confidence = float(np.max(probs))
    emoji = GESTURE_EMOJI[winner]
    return f"""
    <div class="result pop">
      <div class="result-label">RESULT</div>
      <p class="result-call"><span class="g-emoji">{emoji}</span>The model calls it <b class="g-{winner}">{winner}</b></p>
      <div class="result-conf">{confidence:.1%} confidence</div>
      <div class="probs">{render_probs(probs)}</div>
    </div>"""


# Design lives in static/theme.css — app.py holds logic and markup only.
CSS = (pathlib.Path(__file__).parent / "static" / "theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
      <h1>Rock Paper Scissors Classifier</h1>
      <span class="meta">dicoding bootcamp</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="intro">Upload a photo of a hand sign — a small convolutional network, '
    'trained from scratch, calls it rock, paper, or scissors.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Image — JPG or PNG, max 1 MB",
    type=ALLOWED_TYPES,
)

# A new upload invalidates the previous verdict
file_key = f"{uploaded_file.name}:{uploaded_file.size}" if uploaded_file else None
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state.pop("probs", None)

if uploaded_file is None:
    st.markdown('<div class="notice">No image yet — the result will appear here.</div>', unsafe_allow_html=True)
    too_big = False
else:
    too_big = uploaded_file.size > MAX_UPLOAD_BYTES
    if too_big:
        st.markdown(
            f'<div class="error-note">That file is {uploaded_file.size / 1024 / 1024:.1f} MB — '
            f"the limit is 1 MB. Please upload a smaller image.</div>",
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([5, 6], gap="small")
        with left:
            st.image(uploaded_file.getvalue(), width=250)
            st.markdown(
                f'<div class="caption">{html.escape(uploaded_file.name)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
            if st.button("Predict"):
                with st.spinner("Reading the hand…"):
                    img = load_img(io.BytesIO(uploaded_file.getvalue()), target_size=IMG_SIZE)
                    st.session_state["probs"] = call_the_throw(img)

        with right:
            if "probs" in st.session_state:
                st.markdown(render_result(st.session_state["probs"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="notice" style="margin-top:2.2rem">Press Predict to classify the image.</div>',
                    unsafe_allow_html=True,
                )
