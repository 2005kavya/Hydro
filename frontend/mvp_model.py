from __future__ import annotations

from typing import Dict, Any, List

import numpy as np


def _norm(value: float, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def compute_hydration_score(face: Dict[str, Any], audio: Dict[str, Any]) -> float:
    lip_s = face.get("lip_saturation")  # 0..255-ish
    skin_s = face.get("skin_saturation")
    eye_redness = face.get("eye_redness")  # ~0..3

    lip_s_norm = _norm(lip_s or 0.0, 30.0, 180.0)
    skin_s_norm = _norm(skin_s or 0.0, 30.0, 160.0)
    eye_red_norm = _norm(eye_redness or 0.0, 0.9, 1.6)  # higher = redder

    visual_moisture = 0.6 * lip_s_norm + 0.4 * skin_s_norm
    visual_dryness = (1.0 - visual_moisture) * 0.7 + eye_red_norm * 0.3

    # Audio (optional)
    if audio.get("available"):
        zcr = audio.get("zcr_mean")
        spec_centroid = audio.get("spec_centroid_mean")
        rms = audio.get("rms_mean")
        zcr_norm = _norm(zcr or 0.0, 0.02, 0.15)
        centroid_norm = _norm(spec_centroid or 0.0, 1000.0, 4000.0)
        # rms not used directly to avoid scale mismatch
        audio_dryness = 0.5 * zcr_norm + 0.5 * centroid_norm
        w_face, w_audio = 0.6, 0.4
    else:
        audio_dryness = 0.0
        w_face, w_audio = 1.0, 0.0

    dryness = (w_face * visual_dryness + w_audio * audio_dryness) / (w_face + w_audio)
    hydration_score = 100.0 * (1.0 - dryness)
    return float(np.clip(hydration_score, 0.0, 100.0))


def build_recommendations(score: float, face: Dict[str, Any], audio: Dict[str, Any]) -> List[str]:
    msgs: List[str] = []

    if score >= 75:
        msgs.append("Hydration looks good. Keep sipping regularly.")
    elif score >= 50:
        msgs.append("Slight dryness hints. Have a glass of water soon.")
    elif score >= 25:
        msgs.append("Low hydration signs detected. Drink water now.")
    else:
        msgs.append("Very low hydration signals. Hydrate ASAP and reassess.")

    lip_s = face.get("lip_saturation")
    skin_s = face.get("skin_saturation")
    eye_red = face.get("eye_redness")

    if lip_s is not None and lip_s < 60:
        msgs.append("Lips look dry. Suggest sipping some water.")
    if skin_s is not None and skin_s < 60:
        msgs.append("Skin looks a bit dry; consistent hydration helps.")
    if eye_red is not None and eye_red > 1.2:
        msgs.append("Eyes appear red; take screen breaks and drink water.")

    if not face.get("face_detected", 0):
        msgs.append("Face not clearly detected. Retake selfie with better lighting.")

    if audio.get("available") and audio.get("duration_sec", 0.0) < 1.0:
        msgs.append("Voice sample is very short; use a 2–4s clip for better results.")

    return msgs

