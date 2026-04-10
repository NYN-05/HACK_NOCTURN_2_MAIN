import {
      Legend,
      PolarAngleAxis,
      PolarGrid,
      PolarRadiusAxis,
      Radar,
      RadarChart,
      ResponsiveContainer,
      Tooltip,
} from 'recharts';
import { VerificationResponse } from '../types';

interface ModelRadarChartProps {
      result: VerificationResponse;
}

export function ModelRadarChart({ result }: ModelRadarChartProps) {
      const isLayerAvailable = (status: string): boolean => {
            const normalizedStatus = status.toLowerCase();
            return normalizedStatus === 'loaded' || normalizedStatus === 'ok' || normalizedStatus === 'active' || normalizedStatus === 'ready';
      };

      const chartData = [
            {
                  model: 'CNN',
                  score: isLayerAvailable(result.layer_status.cnn) ? Number(result.layer_scores.cnn.toFixed(1)) : 0,
                  reliability: isLayerAvailable(result.layer_status.cnn) ? Number((result.layer_reliabilities.cnn * 100).toFixed(1)) : 0,
            },
            {
                  model: 'VIT',
                  score: isLayerAvailable(result.layer_status.vit) ? Number(result.layer_scores.vit.toFixed(1)) : 0,
                  reliability: isLayerAvailable(result.layer_status.vit) ? Number((result.layer_reliabilities.vit * 100).toFixed(1)) : 0,
            },
            {
                  model: 'GAN',
                  score: isLayerAvailable(result.layer_status.gan) ? Number(result.layer_scores.gan.toFixed(1)) : 0,
                  reliability: isLayerAvailable(result.layer_status.gan) ? Number((result.layer_reliabilities.gan * 100).toFixed(1)) : 0,
            },
            {
                  model: 'OCR',
                  score: isLayerAvailable(result.layer_status.ocr) ? Number(result.layer_scores.ocr.toFixed(1)) : 0,
                  reliability: isLayerAvailable(result.layer_status.ocr) ? Number((result.layer_reliabilities.ocr * 100).toFixed(1)) : 0,
            },
      ];

      return (
            <div className="h-full w-full">
                  <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={chartData} outerRadius="70%">
                              <PolarGrid stroke="rgba(123,97,255,0.35)" />
                              <PolarAngleAxis dataKey="model" tick={{ fill: '#E7DEFF', fontSize: 11, fontWeight: 600 }} />
                              <PolarRadiusAxis
                                    angle={30}
                                    domain={[0, 100]}
                                    tick={{ fill: 'rgba(231,222,255,0.65)', fontSize: 10 }}
                                    tickCount={5}
                              />
                              <Radar
                                    name="Score"
                                    dataKey="score"
                                    stroke="#E91E8C"
                                    fill="#E91E8C"
                                    fillOpacity={0.28}
                                    strokeWidth={2}
                              />
                              <Radar
                                    name="Reliability"
                                    dataKey="reliability"
                                    stroke="#00E5FF"
                                    fill="#00E5FF"
                                    fillOpacity={0.28}
                                    strokeWidth={2}
                              />
                              <Tooltip
                                    contentStyle={{
                                          background: '#0B0B0B',
                                          border: '1px solid rgba(123,97,255,0.45)',
                                          borderRadius: '10px',
                                          color: '#F3ECFF',
                                    }}
                                    formatter={(value) => `${value ?? 0}%`}
                              />
                              <Legend wrapperStyle={{ color: '#E7DEFF', fontSize: 11 }} />
                        </RadarChart>
                  </ResponsiveContainer>
            </div>
      );
}
