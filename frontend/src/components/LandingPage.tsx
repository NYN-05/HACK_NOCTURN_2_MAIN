import { Button } from './Button';
import { Card } from './Card';

interface LandingPageProps {
  endpointLabel: string;
  onEnterWorkspace: () => void;
  onJumpToUpload: () => void;
}

const proofPoints = [
  {
    title: 'Fast review',
    description: 'Preview the file locally, then send a single verification request only when the operator is ready.',
  },
  {
    title: 'Layered evidence',
    description: 'The result shows score, confidence, and supporting layer signals without burying the main verdict.',
  },
  {
    title: 'Low cognitive load',
    description: 'One landing page, one action page, no unnecessary navigation clutter.',
  },
];

export function LandingPage({ endpointLabel, onEnterWorkspace, onJumpToUpload }: LandingPageProps) {
  return (
    <main className="landing-page">
      <section className="landing-hero card">
        <div className="landing-copy">
          <p className="eyebrow">AI image verification</p>
          <h1>One upload. One verdict. Layered evidence.</h1>
          <p className="landing-lede">
            VeriSight gives operators a fast, explainable way to review image authenticity without forcing them through
            a cluttered dashboard.
          </p>

          <div className="landing-actions">
            <Button onClick={onEnterWorkspace}>Start verification</Button>
            <Button type="button" variant="ghost" onClick={onJumpToUpload}>See upload flow</Button>
          </div>

          <dl className="landing-proof-strip">
            <div>
              <dt>Endpoint</dt>
              <dd>{endpointLabel}</dd>
            </div>
            <div>
              <dt>Flow</dt>
              <dd>Preview first, analyze second</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>Score, decision, confidence</dd>
            </div>
          </dl>
        </div>

        <Card className="landing-sidecard">
          <p className="eyebrow">How it works</p>
          <div className="landing-steps">
            {proofPoints.map((point, index) => (
              <article key={point.title} className="landing-step">
                <span>{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <h2>{point.title}</h2>
                  <p>{point.description}</p>
                </div>
              </article>
            ))}
          </div>
        </Card>
      </section>
    </main>
  );
}
