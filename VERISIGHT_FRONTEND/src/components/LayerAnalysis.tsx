import { LAYER_CONFIG } from '../config';
import { VerificationResponse } from '../types';

interface LayerAnalysisProps {
      response: VerificationResponse;
}

export function LayerAnalysis({ response }: LayerAnalysisProps) {
      const layers = ['cnn', 'vit', 'gan', 'ocr'] as const;

      const isLayerAvailable = (status: string): boolean => {
            const normalizedStatus = status.toLowerCase();
            return normalizedStatus === 'loaded' || normalizedStatus === 'ok' || normalizedStatus === 'active' || normalizedStatus === 'ready';
      };

      const getScoreBgColor = (score: number): string => {
            if (score >= 80) return 'bg-green-50 border-green-200';
            if (score >= 60) return 'bg-yellow-50 border-yellow-200';
            if (score >= 40) return 'bg-orange-50 border-orange-200';
            return 'bg-red-50 border-red-200';
      };

      const getScoreBarColor = (score: number): string => {
            if (score >= 80) return 'bg-success';
            if (score >= 60) return 'bg-warning';
            if (score >= 40) return 'bg-alert';
            return 'bg-danger';
      };

      const getReliabilityText = (reliability: number): string => {
            if (reliability >= 0.9) return 'Very High';
            if (reliability >= 0.75) return 'High';
            if (reliability >= 0.6) return 'Medium';
            return 'Low';
      };

      return (
            <div className="w-full max-w-4xl mx-auto mt-8">
                  <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6">
                        <h3 className="text-lg font-bold text-slate-900 mb-6">AI Layer Analysis</h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {layers.map((layer) => {
                                    const status = response.layer_status[layer];
                                    const score = response.layer_scores[layer];
                                    const reliability = response.layer_reliabilities[layer];
                                    const weight = response.effective_weights[layer];
                                    const config = LAYER_CONFIG[layer];

                                    const isLoaded = isLayerAvailable(status);

                                    return (
                                          <div
                                                key={layer}
                                                className={`border rounded-lg p-4 transition-all duration-300 ${getScoreBgColor(score)}`}
                                          >
                                                <div className="flex justify-between items-start mb-3">
                                                      <div>
                                                            <p className="font-bold text-slate-900">{config.name}</p>
                                                            <p className="text-xs text-slate-600 mt-1">{config.fullName}</p>
                                                      </div>
                                                      {isLoaded && (
                                                            <div className="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded">
                                                                  Active
                                                            </div>
                                                      )}
                                                      {status === 'error' && (
                                                            <div className="bg-red-100 text-red-800 text-xs font-bold px-2 py-1 rounded">
                                                                  Error
                                                            </div>
                                                      )}
                                                </div>

                                                {isLoaded ? (
                                                      <div className="space-y-3">
                                                            <div>
                                                                  <div className="flex justify-between items-center mb-2">
                                                                        <span className="text-sm font-semibold text-slate-700">Score</span>
                                                                        <span className="text-lg font-bold" style={{ color: getScoreBarColor(score) === 'bg-success' ? '#22c55e' : 'inherit' }}>
                                                                              {score.toFixed(1)}
                                                                        </span>
                                                                  </div>
                                                                  <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                                                                        <div
                                                                              className={`h-full transition-all duration-500 ${getScoreBarColor(score)}`}
                                                                              style={{ width: `${Math.min(score, 100)}%` }}
                                                                        />
                                                                  </div>
                                                            </div>

                                                            <div className="grid grid-cols-2 gap-3 text-xs">
                                                                  <div className="bg-white/60 rounded p-2">
                                                                        <p className="text-slate-600">Reliability</p>
                                                                        <p className="font-semibold text-slate-900">{getReliabilityText(reliability)}</p>
                                                                        <p className="text-slate-500">{(reliability * 100).toFixed(0)}%</p>
                                                                  </div>
                                                                  <div className="bg-white/60 rounded p-2">
                                                                        <p className="text-slate-600">Weight</p>
                                                                        <p className="font-semibold text-slate-900">{(weight * 100).toFixed(1)}%</p>
                                                                        <p className="text-slate-500">of total</p>
                                                                  </div>
                                                            </div>

                                                            <p className="text-xs text-slate-600 italic">{config.description}</p>
                                                      </div>
                                                ) : (
                                                      <p className="text-sm text-slate-500 py-4">Layer unavailable for this request</p>
                                                )}
                                          </div>
                                    );
                              })}
                        </div>

                        <div className="mt-6 pt-4 border-t border-slate-200">
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="text-center">
                                          <p className="text-xs text-slate-600 uppercase tracking-wide">Fusion Strategy</p>
                                          <p className="font-semibold text-slate-900 text-sm mt-1 capitalize">
                                                {response.fusion_strategy.replace(/_/g, ' ')}
                                          </p>
                                    </div>
                                    <div className="text-center">
                                          <p className="text-xs text-slate-600 uppercase tracking-wide">Processing Time</p>
                                          <p className="font-semibold text-slate-900 text-sm mt-1">
                                                {(response.processing_time_ms / 1000).toFixed(2)}s
                                          </p>
                                    </div>
                                    <div className="text-center">
                                          <p className="text-xs text-slate-600 uppercase tracking-wide">Meta Model</p>
                                          <p className="font-semibold text-slate-900 text-sm mt-1">
                                                {response.meta_model_used ? 'Active' : 'Inactive'}
                                          </p>
                                    </div>
                                    <div className="text-center">
                                          <p className="text-xs text-slate-600 uppercase tracking-wide">Early Exit</p>
                                          <p className="font-semibold text-slate-900 text-sm mt-1">
                                                {response.early_exit_triggered ? 'Yes' : 'No'}
                                          </p>
                                    </div>
                              </div>
                        </div>
                  </div>
            </div>
      );
}
