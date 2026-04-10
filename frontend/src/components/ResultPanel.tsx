import type { VerificationResult } from '../types';
import { Card } from './Card';
import { SectionTitle } from './SectionTitle';

interface ResultPanelProps {
  result: VerificationResult | null;
  error: string | null;
  isLoading: boolean;
}

function decisionLabel(result: VerificationResult): string {
  if (result.decision === 'Verified authentic') {
    return 'VERIFIED';
  }

  if (result.decision === 'Likely authentic') {
    return 'LIKELY REAL';
  }

  if (result.decision === 'Likely manipulated') {
    return 'FLAGGED';
  }

  return 'REVIEW';
}

function decisionTone(result: VerificationResult): 'success' | 'warning' | 'danger' {
  if (result.decision === 'Verified authentic') {
    return 'success';
  }

  if (result.decision === 'Likely manipulated') {
    return 'danger';
  }

  return 'warning';
}

function confidenceTone(score: number): 'success' | 'warning' | 'danger' {
  if (score >= 80) {
    return 'success';
  }

  if (score >= 55) {
    return 'warning';
  }

  return 'danger';
}

export function ResultPanel({ result, error, isLoading }: ResultPanelProps) {
  if (error && !result) {
    return (
      <Card className="result-panel result-panel--error">
        <SectionTitle eyebrow="Result" title="Verification could not complete" description={error} />
        <p className="result-panel__note">Check the endpoint configuration or retry once the service is available.</p>
      </Card>
    );
  }

  if (isLoading && !result) {
    return (
      <Card className="result-panel result-panel--loading">
        <SectionTitle eyebrow="Result" title="Analyzing image" description="Uploading the file and waiting for the verification response." />
        <div className="loading-skeleton" aria-hidden="true">
          <div className="loading-skeleton__bar loading-skeleton__bar--wide" />
          <div className="loading-skeleton__bar" />
          <div className="loading-skeleton__bar" />
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="result-panel result-panel--empty">
        <SectionTitle eyebrow="Result" title="No verification result yet" description="Upload an image and run verification to see the score, decision, confidence, and layer breakdown here." />
      </Card>
    );
  }

  const verdict = decisionLabel(result);
  const verdictTone = decisionTone(result);
  const confidenceAccent = confidenceTone(result.confidence);

  return (
    <Card className="result-panel">
      <SectionTitle
        eyebrow="Result"
        title="Verification outcome"
        description="The decision is shown first, with confidence, processing time, and layer evidence as supporting context."
      />

      <div className="result-panel__hero">
        <div className={`result-verdict result-verdict--${verdictTone}`}>
          <span>Decision</span>
          <strong>{verdict}</strong>
        </div>

        <div className="result-score" aria-label={`Authenticity score ${result.score} out of 100`}>
          <strong>{result.score}</strong>
          <span>score</span>
        </div>
      </div>

      <div className="result-panel__meta">
        <div className={`result-confidence result-confidence--${confidenceAccent}`}>
          <span>Confidence</span>
          <strong>{result.confidence}%</strong>
          <progress value={result.confidence} max={100} aria-label={`Confidence ${result.confidence}%`} />
        </div>
        <div>
          <span>Processing</span>
          <strong>{result.processingMs} ms</strong>
        </div>
        <div>
          <span>Model</span>
          <strong>{result.modelName}</strong>
        </div>
        <div>
          <span>Checked</span>
          <strong>{new Date(result.checkedAt).toLocaleString()}</strong>
        </div>
      </div>

      <p className="result-panel__summary">{result.summary}</p>

      <div className="layer-list" aria-label="Layer breakdown">
        {result.layerBreakdown.map((layer) => (
          <article key={layer.name} className="layer-item">
            <div className="layer-item__head">
              <div>
                <strong>{layer.name}</strong>
                <p>{layer.note}</p>
              </div>
              <span>{layer.score}%</span>
            </div>
            <progress value={layer.score} max={100} aria-label={`${layer.name} score ${layer.score}%`} />
          </article>
        ))}
      </div>
    </Card>
  );
}
