export type VerificationDecision =
  | 'Verified authentic'
  | 'Likely authentic'
  | 'Needs review'
  | 'Likely manipulated';

export type WorkspaceView = 'landing' | 'workspace';

export interface LayerInsight {
  name: string;
  score: number;
  weight: number;
  note: string;
}

export interface VerificationResult {
  score: number;
  decision: VerificationDecision;
  confidence: number;
  summary: string;
  processingMs: number;
  modelName: string;
  checkedAt: string;
  layerBreakdown: LayerInsight[];
}

export interface RunHistoryItem extends VerificationResult {
  fileName: string;
  fileSize: number;
}
