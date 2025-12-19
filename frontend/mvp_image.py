from __future__ import annotations

from typing import Dict, Any, Tuple

import cv2
import numpy as np


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image bytes")
    return image_bgr


def _polygon_mask(image_shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = image_shape
    x, y, rw, rh = rect
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)
    return mask


def _hsv_means(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0, 0.0, 0.0
    region = hsv[ys, xs, :]
    return float(np.mean(region[:, 0])), float(np.mean(region[:, 1])), float(np.mean(region[:, 2]))


def _redness_ratio(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    eps = 1e-6
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    region = image_bgr[ys, xs, :].astype(np.float32)
    r = region[:, 2]
    g = region[:, 1]
    b = region[:, 0]
    return float(np.mean(np.clip(r / (g + b + eps), 0.0, 3.0)))


def extract_face_features(image_bytes: bytes) -> Dict[str, Any]:
    image_bgr = _bytes_to_bgr(image_bytes)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    features: Dict[str, Any] = {
        "face_detected": 0,
        "lip_saturation": None,
        "eye_redness": None,
        "skin_saturation": None,
    }

    if len(faces) == 0:
        return features

    features["face_detected"] = 1
    x, y, w, h = faces[0]

    # Define regions relative to face rectangle
    lip_rect = (x + int(0.2 * w), y + int(0.65 * h), int(0.6 * w), int(0.25 * h))
    eye_rect = (x + int(0.15 * w), y + int(0.2 * h), int(0.7 * w), int(0.25 * h))
    cheek_left = (x + int(0.15 * w), y + int(0.5 * h), int(0.2 * w), int(0.2 * h))
    cheek_right = (x + int(0.65 * w), y + int(0.5 * h), int(0.2 * w), int(0.2 * h))

    lip_mask = _polygon_mask(gray.shape, lip_rect)
    eye_mask = _polygon_mask(gray.shape, eye_rect)
    cheek_mask = cv2.bitwise_or(_polygon_mask(gray.shape, cheek_left), _polygon_mask(gray.shape, cheek_right))

    # Lip saturation (HSV S)
    _, lip_s, _ = _hsv_means(image_bgr, lip_mask)
    features["lip_saturation"] = round(float(lip_s), 3)

    # Eye redness proxy
    eye_red = _redness_ratio(image_bgr, eye_mask)
    features["eye_redness"] = round(float(eye_red), 3)

    # Skin saturation on cheeks
    _, skin_s, _ = _hsv_means(image_bgr, cheek_mask)
    features["skin_saturation"] = round(float(skin_s), 3)

    return features

