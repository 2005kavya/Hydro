import io
from typing import Optional, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image

from mvp_image import extract_face_features
from mvp_audio import extract_audio_features
from mvp_model import compute_hydration_score, build_recommendations


st.set_page_config(page_title="HydroAlert MVP", page_icon="💧", layout="centered")

st.title("HydroAlert MVP 💧 – Face + Voice Hydration Signals")
st.write("All processing is local. Media never leaves your device.")

with st.expander("What this MVP does", expanded=False):
    st.markdown(
        "- Face: lip dryness proxy, eye redness proxy, cheek saturation proxy.\n"
        "- Voice: MFCC summary, spectral features (if libraries available).\n"
        "- Heuristic model → 0–100 hydration score."
    )

st.subheader("1) Capture a selfie")
img_file = st.camera_input("Take a selfie")

st.subheader("2) Add a short voice sample (optional)")
audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], accept_multiple_files=False)

analyze = st.button("Analyze")


def read_image_bytes(uploaded_image) -> Optional[bytes]:
    if uploaded_image is None:
        return None
    return uploaded_image.getvalue()


def read_audio_bytes(uploaded_audio) -> Optional[bytes]:
    if uploaded_audio is None:
        return None
    return uploaded_audio.getvalue()


if analyze:
    face_features: Dict[str, Any] = {}
    audio_features: Dict[str, Any] = {}

    # Face features
    img_bytes = read_image_bytes(img_file)
    if img_bytes:
        with st.spinner("Analyzing face features..."):
            try:
                face_features = extract_face_features(img_bytes)
            except Exception as e:
                st.warning(f"Face analysis failed: {e}")
    else:
        st.info("Please capture a selfie to enable face analysis.")

    # Audio features
    audio_bytes = read_audio_bytes(audio_file)
    if audio_bytes:
        with st.spinner("Analyzing voice features..."):
            try:
                audio_features = extract_audio_features(audio_bytes, filename=audio_file.name)
            except Exception as e:
                st.warning(f"Audio analysis failed: {e}")
    else:
        st.caption("No audio provided; continuing with face-only analysis.")

    # Score & recommendations
    with st.spinner("Computing hydration score..."):
        score = compute_hydration_score(face_features, audio_features)
        recommendations = build_recommendations(score, face_features, audio_features)

    st.divider()
    st.subheader("Results")
    st.metric("Hydration score", f"{int(round(score))}/100")

    if recommendations:
        for msg in recommendations:
            st.write(f"- {msg}")

    with st.expander("See extracted features"):
        st.json({
            "face": face_features,
            "audio": audio_features,
        })

st.caption("Prototype – not a medical device.")
