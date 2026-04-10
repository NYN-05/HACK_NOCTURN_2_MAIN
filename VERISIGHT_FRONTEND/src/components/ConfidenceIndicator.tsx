import { formatConfidence } from '../utils/helpers';

interface ConfidenceIndicatorProps {
      confidence: number;
}

export function ConfidenceIndicator({ confidence }: ConfidenceIndicatorProps) {
      const percentage = formatConfidence(confidence);

      const getConfidenceColor = (conf: number): string => {
            if (conf >= 0.9) return 'from-green-400 to-green-600';
            if (conf >= 0.75) return 'from-yellow-400 to-yellow-600';
            if (conf >= 0.6) return 'from-orange-400 to-orange-600';
            return 'from-red-400 to-red-600';
      };

      const getConfidenceLevel = (conf: number): string => {
            if (conf >= 0.9) return 'Very High';
            if (conf >= 0.75) return 'High';
            if (conf >= 0.6) return 'Medium';
            return 'Low';
      };

      return (
            <div className="w-full max-w-2xl mx-auto mt-6">
                  <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6">
                        <div className="flex items-center justify-between mb-4">
                              <h4 className="text-sm font-bold text-slate-700">Model Confidence</h4>
                              <span className="text-sm font-bold text-slate-600">{percentage}%</span>
                        </div>

                        <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden mb-3">
                              <div
                                    className={`h-full transition-all duration-500 bg-gradient-to-r ${getConfidenceColor(confidence)}`}
                                    style={{ width: `${percentage}%` }}
                              />
                        </div>

                        <p className="text-xs text-slate-600">
                              <span className="font-semibold">{getConfidenceLevel(confidence)} confidence</span> in this analysis
                        </p>
                  </div>
            </div>
      );
}
