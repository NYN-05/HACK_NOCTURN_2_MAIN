import { DECISION_CONFIG } from '../config';

interface DecisionBadgeProps {
      decision: 'AUTO_APPROVE' | 'FAST_TRACK' | 'SUSPICIOUS' | 'REJECT' | 'INCONCLUSIVE';
      score?: number;
      abstained?: boolean;
}

export function DecisionBadge({ decision, score, abstained }: DecisionBadgeProps) {
      const config = DECISION_CONFIG[decision as keyof typeof DECISION_CONFIG];

      if (!config) {
            return null;
      }

      return (
            <div
                  className="rounded-xl p-6 border-2 text-center transition-all duration-300"
                  style={{
                        backgroundColor: config.bg,
                        borderColor: config.borderColor,
                  }}
            >
                  <h3
                        className="text-3xl font-bold mb-2"
                        style={{ color: config.color }}
                  >
                        {config.label}
                  </h3>

                  {abstained ? (
                        <>
                              <p className="text-5xl font-bold text-slate-400 mb-2">--</p>
                              <p className="text-slate-600 text-sm">Inconclusive Analysis</p>
                        </>
                  ) : (
                        <>
                              <p className="text-5xl font-bold mb-2" style={{ color: config.color }}>
                                    {score}
                              </p>
                              <p className="text-slate-600 text-sm">Authenticity Score</p>
                        </>
                  )}

                  <p className="mt-4 text-slate-700 text-sm font-medium">{config.description}</p>
            </div>
      );
}
