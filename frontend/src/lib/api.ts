import type { LayerInsight, VerificationDecision, VerificationResult } from '../types';

const DEFAULT_ENDPOINT = '/api/v1/verify';
const LAYER_ORDER = ['cnn', 'vit', 'gan', 'ocr'] as const;

type LayerKey = (typeof LAYER_ORDER)[number];
type BackendDecision = 'AUTO_APPROVE' | 'FAST_TRACK' | 'SUSPICIOUS' | 'REJECT' | 'ABSTAIN';

type BackendLayerPayload = {
  score?: number;
};

type BackendResult = Partial<VerificationResult> & {
  authenticity_score?: number;
  confidenceScore?: number;
  inferenceMs?: number;
  model?: string;
  processing_time_ms?: number;
  decision?: BackendDecision | VerificationDecision | string;
  layer_scores?: Partial<Record<LayerKey, number>>;
  effective_weights?: Partial<Record<LayerKey, number>>;
  layer_reliabilities?: Partial<Record<LayerKey, number>>;
  layer_status?: Partial<Record<LayerKey, string>>;
  layer_outputs?: Partial<Record<LayerKey, BackendLayerPayload>>;
};

export function getVerificationEndpointLabel(): string {
  const baseUrl = (import.meta.env.VITE_VERIFY_API_URL ?? DEFAULT_ENDPOINT).replace(/\/$/, '');
  return baseUrl.endsWith('/verify') ? baseUrl : `${baseUrl}/verify`;
}

const LAYER_DETAILS: Record<LayerKey, Omit<LayerInsight, 'score'>> = {
  cnn: {
    name: 'Layer 1 - CNN + ELA',
    weight: 41,
    note: 'Checks pixel-level tampering traces, local texture changes, and compression residue.',
  },
  vit: {
    name: 'Layer 2 - Vision Transformer',
    weight: 8,
    note: 'Checks whole-image structure and whether the scene still makes semantic sense overall.',
  },
  gan: {
    name: 'Layer 3 - GAN detector',
    weight: 41,
    note: 'Looks for generator-style artifacts and synthetic image patterns missed by simple pixel checks.',
  },
  ocr: {
    name: 'Layer 4 - OCR validation',
    weight: 10,
    note: 'Reads visible text such as expiry details and checks whether it looks plausible.',
  },
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toNumber(value: unknown, fallback: number): number {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : fallback;
}

function formatDecision(score: number): VerificationDecision {
  if (score >= 88) {
    return 'Verified authentic';
  }

  if (score >= 64) {
    return 'Likely authentic';
  }

  if (score >= 44) {
    return 'Needs review';
  }

  return 'Likely manipulated';
}

function mapDecision(decision: BackendDecision | VerificationDecision | string | undefined, score: number): VerificationDecision {
  if (decision === 'Verified authentic' || decision === 'Likely authentic' || decision === 'Needs review' || decision === 'Likely manipulated') {
    return decision;
  }

  switch (decision) {
    case 'AUTO_APPROVE':
      return 'Verified authentic';
    case 'FAST_TRACK':
      return 'Likely authentic';
    case 'SUSPICIOUS':
    case 'ABSTAIN':
      return 'Needs review';
    case 'REJECT':
      return 'Likely manipulated';
    default:
      return formatDecision(score);
  }
}

function buildSummary(decision: VerificationDecision): string {
  switch (decision) {
    case 'Verified authentic':
      return 'The engine found strong authenticity evidence across the available layers.';
    case 'Likely authentic':
      return 'Most signals look authentic, but the result is not strong enough for an automatic approval.';
    case 'Needs review':
      return 'The layers disagree or remain uncertain, so a manual review is recommended.';
    case 'Likely manipulated':
      return 'Multiple signals suggest the image may be edited, synthetic, or otherwise unreliable.';
    default:
      return 'The verification result is available for review.';
  }
}

function buildFallbackLayerBreakdown(seed: number, score: number): LayerInsight[] {
  return LAYER_ORDER.map((key, index) => {
    const offset = (seed * (index + 3) + index * 17) % 28;
    const layerScore = clamp(Math.round(score + offset - 14), 5, 95);
    const layer = LAYER_DETAILS[key];

    return {
      name: layer.name,
      score: layerScore,
      weight: layer.weight,
      note: layer.note,
    };
  });
}

export function buildFallbackResult(file: File): VerificationResult {
  const seed = [...file.name].reduce((total, char) => total + char.charCodeAt(0), file.size % 97);
  const score = clamp(Math.round((seed % 100) * 0.72 + (file.type.startsWith('image/') ? 14 : 0)), 4, 96);
  const confidence = clamp(Math.round(64 + (seed % 31)), 55, 98);
  const decision = formatDecision(score);
  const layerBreakdown = buildFallbackLayerBreakdown(seed, score);
  const processingMs = 640 + (seed % 760);

  return {
    score,
    decision,
    confidence,
    summary: buildSummary(decision),
    processingMs,
    modelName: 'VeriSight Orchestrator (mock)',
    checkedAt: new Date().toISOString(),
    layerBreakdown,
  };
}

function describeLayerNote(baseNote: string, status?: string, reliability?: number): string {
  if (status === 'skipped') {
    return `${baseNote} Skipped because an earlier layer already produced a decisive result.`;
  }

  if (status === 'degraded') {
    return `${baseNote} Ran in degraded mode, so the engine reduced its influence on the final score.`;
  }

  if (typeof reliability === 'number' && Number.isFinite(reliability)) {
    return `${baseNote} Reliability ${Math.round(clamp(reliability, 0, 1) * 100)}%.`;
  }

  return baseNote;
}

function buildLayerBreakdown(candidate: BackendResult, fallback: VerificationResult): LayerInsight[] {
  if (Array.isArray(candidate.layerBreakdown) && candidate.layerBreakdown.length > 0) {
    return candidate.layerBreakdown.map((entry) => ({
      name: String((entry as LayerInsight).name ?? 'Layer'),
      score: clamp(toNumber((entry as LayerInsight).score, 0), 0, 100),
      weight: clamp(toNumber((entry as LayerInsight).weight, 0), 0, 100),
      note: String((entry as LayerInsight).note ?? ''),
    }));
  }

  const keys = LAYER_ORDER.filter(
    (key) =>
      candidate.layer_scores?.[key] !== undefined ||
      candidate.layer_status?.[key] !== undefined ||
      candidate.layer_outputs?.[key] !== undefined,
  );

  if (keys.length === 0) {
    return fallback.layerBreakdown;
  }

  return keys.map((key) => {
    const layer = LAYER_DETAILS[key];
    const score = clamp(
      toNumber(candidate.layer_scores?.[key] ?? candidate.layer_outputs?.[key]?.score, 50),
      0,
      100,
    );
    const effectiveWeight = candidate.effective_weights?.[key];
    const reliability = candidate.layer_reliabilities?.[key];

    return {
      name: layer.name,
      score,
      weight: clamp(
        Math.round(typeof effectiveWeight === 'number' ? effectiveWeight * 100 : layer.weight),
        0,
        100,
      ),
      note: describeLayerNote(layer.note, candidate.layer_status?.[key], reliability),
    };
  });
}

function normalizeResult(data: unknown, fallback: VerificationResult): VerificationResult {
  if (!data || typeof data !== 'object') {
    return fallback;
  }

  const candidate = data as BackendResult;
  const score = clamp(toNumber(candidate.authenticity_score ?? candidate.score, fallback.score), 0, 100);
  const rawConfidence = toNumber(candidate.confidence ?? candidate.confidenceScore, fallback.confidence);
  const confidence = clamp(rawConfidence <= 1 ? Math.round(rawConfidence * 100) : rawConfidence, 0, 100);
  const decision = mapDecision(candidate.decision, score);
  const layerBreakdown = buildLayerBreakdown(candidate, fallback);

  return {
    score,
    decision,
    confidence,
    summary: typeof candidate.summary === 'string' && candidate.summary.trim() ? candidate.summary : buildSummary(decision),
    processingMs: toNumber(candidate.processingMs ?? candidate.processing_time_ms ?? candidate.inferenceMs, fallback.processingMs),
    modelName:
      typeof candidate.modelName === 'string' && candidate.modelName.trim()
        ? candidate.modelName
        : typeof candidate.model === 'string' && candidate.model.trim()
          ? candidate.model
          : 'VeriSight Orchestrator',
    checkedAt: typeof candidate.checkedAt === 'string' ? candidate.checkedAt : fallback.checkedAt,
    layerBreakdown,
  };
}

export async function verifyImage(file: File): Promise<VerificationResult> {
  const fallback = buildFallbackResult(file);
  const useMock = import.meta.env.VITE_USE_MOCK_API === 'true';
  const endpoint = getVerificationEndpointLabel();

  if (useMock) {
    return fallback;
  }

  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let message = `Verification request failed with ${response.status}`;

    try {
      const errorPayload = await response.json();
      if (typeof errorPayload?.detail === 'string' && errorPayload.detail.trim()) {
        message = errorPayload.detail;
      }
    } catch {
      // Fall back to the status-based message when the backend response is not JSON.
    }

    throw new Error(message);
  }

  const data = await response.json();
  return normalizeResult(data, fallback);
}
