import { Suspense, lazy, useState } from 'react';
import { LandingPage } from './components/LandingPage';
import { getVerificationEndpointLabel } from './lib/api';

const WorkspaceShell = lazy(() => import('./components/WorkspaceShell').then((module) => ({ default: module.WorkspaceShell })));

type View = 'landing' | 'workspace';

export default function App() {
  const [view, setView] = useState<View>('landing');
  const endpointLabel = getVerificationEndpointLabel();

  if (view === 'landing') {
    return (
      <LandingPage
        endpointLabel={endpointLabel}
        onEnterWorkspace={() => setView('workspace')}
        onJumpToUpload={() => setView('workspace')}
      />
    );
  }

  return (
    <Suspense fallback={<div className="workspace-page"><div className="card workspace-loading">Loading workspace…</div></div>}>
      <WorkspaceShell onOpenLanding={() => setView('landing')} />
    </Suspense>
  );
}