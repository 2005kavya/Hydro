# HydroAlert MVP (Face + Voice, Local)

Minimal Streamlit MVP that analyzes a selfie and an optional voice clip locally to compute a hydration score.

## Run

```bash
cd frontend
../venv/Scripts/streamlit.exe run streamlit_mvp.py
```

If `librosa` is not installed, audio analysis is skipped automatically.

## Files
- `frontend/streamlit_mvp.py`: Streamlit UI
- `frontend/mvp_image.py`: OpenCV-based face features (lip saturation, eye redness proxy, cheek saturation)
- `frontend/mvp_audio.py`: Optional librosa-based audio features
- `frontend/mvp_model.py`: Heuristic hydration score + recommendations

Note: Prototype only; not a medical device.

