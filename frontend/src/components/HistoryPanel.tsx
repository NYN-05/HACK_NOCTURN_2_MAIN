import type { RunHistoryItem } from '../types';
import { Card } from './Card';
import { SectionTitle } from './SectionTitle';

interface HistoryPanelProps {
  history: RunHistoryItem[];
}

function formatBytes(size: number): string {
  if (size === 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);
  return `${(size / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function HistoryPanel({ history }: HistoryPanelProps) {
  return (
    <Card className="history-panel">
      <SectionTitle
        eyebrow="History"
        title="Recent runs"
        description="The last few analyses stay visible so operators can compare results without re-uploading files."
      />

      {history.length > 0 ? (
        <div className="history-list">
          {history.slice(0, 4).map((item) => (
            <article key={`${item.fileName}-${item.checkedAt}`} className="history-item">
              <div>
                <strong>{item.fileName}</strong>
                <p>{item.decision}</p>
                <span>{formatBytes(item.fileSize)}</span>
              </div>
              <div className="history-item__score">
                <strong>{item.score}%</strong>
                <span>{new Date(item.checkedAt).toLocaleTimeString()}</span>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <p className="history-empty">No runs yet. The latest analysis will appear here.</p>
      )}
    </Card>
  );
}