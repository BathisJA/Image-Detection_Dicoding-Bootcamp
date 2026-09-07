import html
import io
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img

IMG_SIZE = (150, 150)
# Same order Keras used at training time (folders sort alphabetically):
# paper -> 0, rock -> 1, scissors -> 2
CLASSES = ["paper", "rock", "scissors"]

st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    page_icon="✊",
    layout="centered",
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


def call_the_throw(pil_img):
    """Same preprocessing as training: RGB, 150x150, rescaled to 0-1."""
    x = np.asarray(pil_img, dtype=np.float32) / 255.0
    return model.predict(np.expand_dims(x, 0), verbose=0)[0]


# ---------------------------------------------------------------------------
# Look & feel: quiet editorial minimalism. One column, warm white, ink text,
# a single terracotta accent, thin hairlines, generous whitespace.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,340;0,9..144,560;1,9..144,420&family=Hanken+Grotesk:wght@400;500;600&family=Fragment+Mono&display=swap');

:root {
  --bg:      #FAF9F6;
  --card:    #FFFFFF;
  --ink:     #171512;
  --muted:   #8B857A;
  --hair:    #E7E3DA;
  --accent:  #C24D21;
}

.stApp {
  background: var(--bg); color: var(--ink);
  font-family: 'Hanken Grotesk', sans-serif;
}

.stApp [data-testid="stHeader"],
.stApp [data-testid="stFooter"] { display: none; }

.stApp .block-container { max-width: 620px !important; padding: 4.5rem 1.4rem 3rem !important; }

.stApp .fade { animation: fade-up .5s ease both; }
@keyframes fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .stApp * { animation: none !important; transition: none !important; } }

/* header */
.stApp .masthead {
  text-align: center; font-family: 'Fragment Mono', monospace !important;
  font-size: .68rem; letter-spacing: .28em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 2.6rem;
}
.stApp .masthead em { font-style: normal; color: var(--accent); }

/* title */
.stApp h1.title {
  font-family: 'Fraunces', serif !important; font-weight: 560;
  font-size: clamp(2.2rem, 5vw, 3rem); line-height: 1.08;
  letter-spacing: -.01em; text-align: center; color: var(--ink);
  margin: 0 0 1rem;
}
.stApp h1.title em { font-style: italic; font-weight: 420; color: var(--accent); }
.stApp .lede {
  text-align: center; max-width: 26rem; margin: 0 auto 3rem;
  font-size: .95rem; line-height: 1.6; color: var(--muted);
}

/* uploader */
.stApp [data-testid="stWidgetLabel"] p {
  font-family: 'Fragment Mono', monospace !important; font-size: .66rem;
  letter-spacing: .2em; text-transform: uppercase; color: var(--muted);
}
.stApp [data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important; border: 1px solid var(--hair) !important;
  box-shadow: none !important;
}
.stApp [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
.stApp [data-testid="stFileUploaderDropzone"] button {
  border: 1px solid var(--ink) !important; background: transparent !important;
  font-family: 'Fragment Mono', monospace !important; font-size: .68rem !important;
  letter-spacing: .12em; text-transform: uppercase; box-shadow: none !important;
}
.stApp [data-testid="stFileUploaderDropzone"] button:hover { background: var(--ink) !important; color: var(--bg) !important; }
.stApp [data-testid="stFileUploaderDropzone"] svg { fill: var(--ink); }

/* uploaded image */
.stApp [data-testid="stImage"] img {
  border-radius: 6px; display: block; margin: 0 auto;
}
.stApp .caption {
  text-align: center; margin-top: .7rem;
  font-family: 'Fragment Mono', monospace !important; font-size: .64rem;
  letter-spacing: .14em; color: var(--muted); text-transform: uppercase;
}

/* predict button */
.stApp .stButton { text-align: center; }
.stApp .stButton > button {
  background: var(--ink) !important; color: var(--bg) !important;
  border: none !important; border-radius: 999px !important;
  font-family: 'Fragment Mono', monospace !important; font-size: .72rem !important;
  letter-spacing: .18em; text-transform: uppercase;
  padding: .65rem 2.4rem !important;
  box-shadow: none !important;
  transition: opacity .15s ease;
}
.stApp .stButton > button:hover { opacity: .82; }

/* verdict */
.stApp .verdict { text-align: center; margin-top: 3.2rem; }
.stApp .verdict-eyebrow {
  font-family: 'Fragment Mono', monospace !important; font-size: .66rem;
  letter-spacing: .24em; text-transform: uppercase; color: var(--muted);
  margin-bottom: .6rem;
}
.stApp .verdict-call {
  font-family: 'Fraunces', serif !important; font-style: italic; font-weight: 420;
  font-size: 2.7rem; line-height: 1; color: var(--accent); margin: 0;
}
.stApp .verdict-conf {
  margin-top: .7rem; font-size: .85rem; color: var(--muted);
}

/* probabilities */
.stApp .probs { max-width: 22rem; margin: 2.4rem auto 0; display: flex; flex-direction: column; gap: .8rem; }
.stApp .prob-row { display: grid; grid-template-columns: 4.6rem 1fr 3rem; gap: .8rem; align-items: center; }
.stApp .prob-label {
  font-family: 'Fragment Mono', monospace !important; font-size: .66rem;
  letter-spacing: .16em; text-transform: uppercase; color: var(--muted);
}
.stApp .prob-track { height: 2px; background: var(--hair); }
.stApp .prob-fill { height: 100%; background: var(--ink); transition: width .6s ease; }
.stApp .prob-fill.fill-accent { background: var(--accent); }
.stApp .prob-pct {
  font-family: 'Fragment Mono', monospace !important; font-size: .66rem;
  text-align: right; color: var(--ink);
}

.stApp .hint {
  text-align: center; margin-top: 3rem;
  font-family: 'Fragment Mono', monospace !important; font-size: .68rem;
  color: var(--muted);
}

.stApp .rule { border-top: 1px solid var(--hair); margin-top: 3.4rem; }
.stApp .footer {
  display: flex; justify-content: space-between; padding-top: .9rem;
  font-family: 'Fragment Mono', monospace !important; font-size: .62rem;
  letter-spacing: .1em; color: var(--muted); text-transform: uppercase;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="fade">
      <div class="masthead">image classifier &middot; dicoding <em>bootcamp</em></div>
      <h1 class="title">Rock <em>Paper</em> Scissors</h1>
      <p class="lede">Upload a photo of a hand sign. A small convolutional network,
      trained from scratch, will call the throw.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Hand sign", type=["jpg", "jpeg", "png"])

# A new upload invalidates the previous verdict
file_key = f"{uploaded_file.name}:{uploaded_file.size}" if uploaded_file else None
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state.pop("probs", None)

if uploaded_file is None:
    st.markdown('<div class="hint">no image yet</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="height:2.2rem"></div>', unsafe_allow_html=True)
    st.image(uploaded_file.getvalue(), width=300)
    st.markdown(
        f'<div class="caption">{html.escape(uploaded_file.name)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)

    if st.button("Predict"):
        with st.spinner("Reading the hand…"):
            img = load_img(io.BytesIO(uploaded_file.getvalue()), target_size=IMG_SIZE)
            st.session_state["probs"] = call_the_throw(img)

    if "probs" in st.session_state:
        probs = st.session_state["probs"]
        winner = CLASSES[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        rows = ""
        for cls, p in zip(CLASSES, probs):
            fill = "fill-accent" if cls == winner else ""
            rows += f"""
            <div class="prob-row">
              <span class="prob-label">{cls}</span>
              <div class="prob-track"><div class="prob-fill {fill}" style="width:{p * 100:.1f}%"></div></div>
              <span class="prob-pct">{p * 100:.0f}%</span>
            </div>"""

        st.markdown(
            f"""
            <div class="verdict fade">
              <div class="verdict-eyebrow">the model calls it</div>
              <p class="verdict-call">{winner}</p>
              <div class="verdict-conf">{confidence:.1%} confident</div>
              <div class="probs">{rows}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
      <span>convnet &middot; 150×150 rgb</span>
      <span>trained in model.ipynb</span>
    </div>
    """,
    unsafe_allow_html=True,
)
