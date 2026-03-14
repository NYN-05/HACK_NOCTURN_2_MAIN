MODEL_WEIGHTS = {
    "cnn": 0.50,
    "vit": 0.30,
    "gan": 0.12,
    "ocr": 0.08,
}

DECISION_THRESHOLDS = [
    (90, "AUTO_APPROVE"),
    (62, "FAST_TRACK"),
    (45, "SUSPICIOUS"),
    (0, "REJECT"),
]

VERIFY_ENDPOINT = "/api/v1/verify"
