HydroAlert Synthetic Dataset (for PPT/demo)

Files:
- metadata.csv: 20 example rows with plausible features and labels

Columns:
- participant_id: anonymized ID (e.g., P001)
- timestamp: ISO timestamp
- selfie_path: suggested path to selfie image
- voice_path: suggested path to audio file
- usg: urine specific gravity label (gold-standard proxy)
- urine_color: 1–8 scale
- time_since_last_drink_min: minutes
- intake_last_3h_ml: ml
- room_temp_c: ambient temperature in Celsius
- exercise_last_2h: true/false
- lip_saturation: HSV saturation proxy for lips (0–255-ish)
- eye_redness: redness ratio (approx 0–3)
- skin_saturation: cheek region saturation proxy (0–255-ish)
- zcr_mean: zero crossing rate
- spec_centroid_mean: spectral centroid (Hz)
- rms_mean: root-mean-square energy
- label_source: "usg" to indicate label came from USG

Note: Synthetic, illustrative only. Use with clear disclaimer; not real patient data.


