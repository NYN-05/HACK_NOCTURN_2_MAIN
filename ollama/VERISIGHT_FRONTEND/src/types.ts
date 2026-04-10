// API Response Types
export interface VerificationResponse {
      schema_version: string;
      authenticity_score: number;
      decision: 'AUTO_APPROVE' | 'FAST_TRACK' | 'SUSPICIOUS' | 'REJECT';
      confidence: number;
      abstained: boolean;
      fusion_strategy: string;
      meta_model_used: boolean;
      early_exit_triggered: boolean;
      processing_time_ms: number;
      layer_scores: LayerScores;
      layer_reliabilities: LayerReliabilities;
      effective_weights: EffectiveWeights;
      layer_status: LayerStatus;
      layer_outputs: Record<string, unknown>;
      available_layers: string[];
}

export interface LayerScores {
      cnn: number;
      vit: number;
      gan: number;
      ocr: number;
}

export interface LayerReliabilities {
      cnn: number;
      vit: number;
      gan: number;
      ocr: number;
}

export interface EffectiveWeights {
      cnn: number;
      vit: number;
      gan: number;
      ocr: number;
}

export interface LayerStatus {
      cnn: 'loaded' | 'unavailable' | 'error' | 'ok' | 'degraded' | 'skipped';
      vit: 'loaded' | 'unavailable' | 'error' | 'ok' | 'degraded' | 'skipped';
      gan: 'loaded' | 'unavailable' | 'error' | 'ok' | 'degraded' | 'skipped';
      ocr: 'loaded' | 'unavailable' | 'error' | 'ok' | 'degraded' | 'skipped';
}

export interface VerificationRequest {
      image: File;
      order_date?: string;
      delivery_date?: string;
      mfg_date_claimed?: string;
}

// UI State Types
export interface UploadState {
      file: File | null;
      preview: string | null;
      isLoading: boolean;
      progress: number;
      error: string | null;
}

export interface ResultState {
      data: VerificationResponse | null;
      isLoading: boolean;
      error: string | null;
}

// Form Data Types
export interface FormData {
      order_date: string;
      delivery_date: string;
      mfg_date_claimed: string;
}
