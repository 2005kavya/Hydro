from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, Any, Optional

import numpy as np

try:
    import librosa  # type: ignore
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False


def _load_audio_from_bytes(audio_bytes: bytes, filename: Optional[str] = None, sr: int = 22050):
    if not HAVE_LIBROSA:
        raise RuntimeError("librosa not available")
    suffix = os.path.splitext(filename or "temp.wav")[1]
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        y, sr = librosa.load(tmp.name, sr=sr, mono=True)
    return y, sr


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def extract_audio_features(audio_bytes: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    if not HAVE_LIBROSA:
        # Graceful degradation: return empty dict to indicate unavailable audio features
        return {"available": False}

    y, sr = _load_audio_from_bytes(audio_bytes, filename)
    if y.size == 0:
        raise ValueError("Empty audio")

    duration = y.shape[0] / float(sr)

    hop_length = 512
    n_fft = 2048

    import librosa  # local import for linters

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)
    rmse = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features: Dict[str, Any] = {
        "available": True,
        "sample_rate": int(sr),
        "duration_sec": round(duration, 3),
        "zcr_mean": round(_safe_mean(zcr), 6),
        "spec_centroid_mean": round(_safe_mean(spec_centroid), 3),
        "spec_rolloff_mean": round(_safe_mean(spec_rolloff), 3),
        "rms_mean": round(_safe_mean(rmse), 6),
        "mfcc_means": [round(float(v), 4) for v in np.mean(mfcc, axis=1).tolist()],
    }

    return features

