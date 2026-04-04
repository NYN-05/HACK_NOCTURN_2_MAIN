# Optimization Roadmap Status

This file maps the requested optimization roadmap to repository implementation status.

## Implemented in Code

1. Canonical end-to-end benchmark and run history
- `evaluation/evaluate_system.py` now reports accuracy, precision, recall, F1, ECE, and latency p50/p90/p99.
- Supports history append and regression comparison.

2. OCR runtime hardening
- Deterministic OCR engine selection via env vars in `interfaces/ocr_interface.py`.
- Explicit provider/fallback telemetry in `layer4/inference/ocr_verification.py`.

3. Confidence calibration support
- Calibrated threshold defaults in `configs/weights.py` and `engine/decision_engine.py`.
- Threshold calibration utility: `evaluation/calibrate_thresholds.py`.

4. Per-layer reliability weighting
- Dynamic reliability-weighted fusion in `engine/scoring_engine.py`.
- Reliability extraction + effective weights in `engine/pipeline/orchestrator.py`.

5. Timeout + graceful degradation
- Per-layer timeouts and degraded fallback path in `engine/pipeline/orchestrator.py`.

6. API schema versioning
- `schema_version` added to verify response in `engine/pipeline/api_router.py`.

7. Structured telemetry and error taxonomy
- Per-run telemetry and per-layer status/reliability in `engine/pipeline/orchestrator.py`.

8. Concurrency and load controls
- API semaphore and request timeout controls in `engine/pipeline/api_router.py`.

9. Warm-load models at startup
- Startup preload in `engine/pipeline/app.py`.

10. Frontend progressive rendering + confidence/fallback panel
- Added pipeline progress state, confidence band, and warning panel in `frontend/src/App.jsx` + `frontend/src/App.css`.

11. Performance regression automation
- Regression checker script: `evaluation/check_regression.py`.
- GitHub workflow: `.github/workflows/performance-regression.yml`.

12. Deterministic environment lock
- Runtime lock file: `requirements.lock.txt`.

## Implemented as Data/Training Tooling

13. Split hygiene and duplicate checks
- `tools/data/split_hygiene.py`.

14. Hard-negative mining support
- `tools/data/hard_negative_mining.py`.

## Requires Dataset/Training Runs (Project-Specific)

15. Strict split enforcement for all layer datasets
- Use `tools/data/split_hygiene.py` in each dataset pipeline.

16. Class/domain-balanced sampling in all training loops
- Requires updates in layer-specific training code and retraining.

17. Augmentation policy search per layer
- Requires experiment sweeps and model retraining.

18. Batched inference for all model paths
- Requires interface-level batching APIs and evaluation across layer implementations.

19. Mixed precision/quantized export with accuracy gates
- Requires export pipelines and acceptance tests per model.

20. Cache expensive preprocessing keyed by image hash
- Requires cache store choice and invalidation policy by deployment environment.

21. Full docs centralization into single canonical source
- Needs content merge decisions across root/layer READMEs.
