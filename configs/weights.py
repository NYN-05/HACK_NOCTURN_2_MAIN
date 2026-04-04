MODEL_WEIGHTS = {
    "cnn": 0.50,
    "vit": 0.30,
    "gan": 0.12,
    "ocr": 0.08,
}

LAYER_TIMEOUT_MS = {
    "cnn": 3500,
    "vit": 3500,
    "gan": 3500,
    "ocr": 4500,
}

LAYER_RELIABILITY_FLOOR = 0.15
ENABLE_DYNAMIC_RELIABILITY_WEIGHTING = True
ENABLE_META_MODEL_FUSION = True
ENABLE_REQUEST_CACHE = True
REQUEST_CACHE_MAX_ITEMS = 256
ENABLE_EARLY_EXIT = True
EARLY_EXIT_CNN_SCORE_THRESHOLD = 95.0
EARLY_EXIT_MIN_RELIABILITY = 0.90

CALIBRATED_MODEL_WEIGHTS = {
    "cnn": 0.46,
    "vit": 0.28,
    "gan": 0.16,
    "ocr": 0.10,
}

DECISION_THRESHOLDS = [
    (90, "AUTO_APPROVE"),
    (62, "FAST_TRACK"),
    (45, "SUSPICIOUS"),
    (0, "REJECT"),
]

CALIBRATED_DECISION_THRESHOLDS = [
    (88, "AUTO_APPROVE"),
    (64, "FAST_TRACK"),
    (44, "SUSPICIOUS"),
    (0, "REJECT"),
]

RESPONSE_SCHEMA_VERSION = "2.0.0"

BENCHMARK_BUDGET = {
    "min_accuracy": 0.85,
    "max_p90_latency_ms": 5000,
    "max_p99_latency_ms": 8000,
}

VERIFY_ENDPOINT = "/api/v1/verify"
