import { Button } from './Button';
import { Card } from './Card';
import { HistoryPanel } from './HistoryPanel';
import { ResultPanel } from './ResultPanel';
import { SectionTitle } from './SectionTitle';
import { UploadDropzone } from './UploadDropzone';
import { useVerificationWorkspace } from '../hooks/useVerificationWorkspace';

interface WorkspaceShellProps {
  onOpenLanding: () => void;
}

export function WorkspaceShell({ onOpenLanding }: WorkspaceShellProps) {
  const {
    analysisState,
    apiModeLabel,
    clearSelection,
    endpointLabel,
    error,
    file,
    history,
    isLoading,
    handleAnalyze,
    handleSelectFile,
    previewUrl,
    result,
  } = useVerificationWorkspace();

  return (
    <main className="workspace-page">
      <header className="workspace-topbar">
        <button type="button" className="brand-lockup" onClick={onOpenLanding}>
          <span className="brand-mark">V</span>
          <span>
            <strong>VeriSight</strong>
            <small>Image verification workspace</small>
          </span>
        </button>

        <div className="workspace-topbar__meta">
          <span className="badge badge--neutral">{apiModeLabel}</span>
          <span className="workspace-topbar__endpoint">{endpointLabel}</span>
        </div>

        <Button variant="ghost" onClick={onOpenLanding}>
          Back to landing
        </Button>
      </header>

      <section className="workspace-hero card">
        <div>
          <p className="eyebrow">Verification workspace</p>
          <h1>Upload one image and get a verdict with layered evidence.</h1>
          <p className="workspace-lede">
            Preview first, analyze second, and keep the result visible with a low-friction flow designed for operators.
          </p>
        </div>

        <dl className="workspace-facts">
          <div>
            <dt>Status</dt>
            <dd>{analysisState}</dd>
          </div>
          <div>
            <dt>Latest score</dt>
            <dd>{result ? `${result.score}%` : '—'}</dd>
          </div>
          <div>
            <dt>Session runs</dt>
            <dd>{history.length}</dd>
          </div>
        </dl>
      </section>

      <section className="workspace-grid">
        <Card className="workspace-card">
          <SectionTitle
            eyebrow="Upload"
            title="Add an image and inspect it before analysis"
            description="The preview updates immediately so the operator can confirm the file before sending it to the verification service."
          />
          <UploadDropzone
            file={file}
            previewUrl={previewUrl}
            onFileSelect={handleSelectFile}
            onAnalyze={handleAnalyze}
            onClear={clearSelection}
            isLoading={isLoading}
          />
        </Card>

        <ResultPanel error={error} isLoading={isLoading} result={result} />

        <HistoryPanel history={history} />
      </section>
    </main>
  );
}
