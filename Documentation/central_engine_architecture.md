# VeriSight Central Engine Architecture

Key principle: all intelligence is distributed across layers, but all coordination is centralized in the engine.

## Layer Contract (Independent Scorers)
Each layer is an independent scorer that takes an image and returns only its own result:

- input: image_path (+ optional metadata when needed)
- output:
  - score: float in [0, 100] (authenticity score for that layer)
  - raw: layer-specific diagnostics

Layers must not call each other or fuse scores.

## Central Coordinator
`engine/pipeline/orchestrator.py` is the only module that:

- invokes all 4 layer scorers
- enforces timeouts/reliability policy
- fuses scores via `ScoringEngine`
- maps final score to decision via `DecisionEngine`
- returns a unified response for API/frontend

## Data Flow
1. API receives image and metadata.
2. API calls only `VerificationOrchestrator.run(...)`.
3. Orchestrator invokes layers independently:
   - layer1 CNN
   - layer2 ViT
   - layer3 GAN
   - layer4 OCR
4. Orchestrator computes reliability and weighted fusion.
5. Orchestrator outputs final score + decision + telemetry.
6. Frontend consumes this single canonical payload.

## Pseudocode
```python
class VerificationOrchestrator:
    def run(image_path, metadata):
        outputs = {}
        for layer_name, layer_predict in layer_registry:
            outputs[layer_name] = timed_call(layer_predict, image_path, metadata)

        scores = {k: outputs[k]["score"] for k in outputs}
        reliability = {k: compute_reliability(outputs[k]) for k in outputs}

        fused = scoring_engine.fuse(scores, reliability)
        decision = decision_engine.classify(fused.weighted_score)

        return {
            "authenticity_score": fused.weighted_score,
            "decision": decision,
            "layer_scores": fused.layer_scores,
            "layer_reliabilities": fused.layer_reliabilities,
            "effective_weights": fused.effective_weights,
            "confidence": fused.confidence,
            "layer_outputs": outputs,
        }
```

## Implementation Notes
- Layer-4 local wrapper (`layer4/orchestrator.py`) is now a pure layer scorer adapter.
- Global orchestration imports are standardized to `engine.pipeline`.
- No layer module should own final score generation.
