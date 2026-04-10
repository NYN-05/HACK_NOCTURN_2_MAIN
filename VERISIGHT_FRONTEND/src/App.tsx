import { useEffect, useRef, useState } from 'react';
import { Header } from './components/Header';
import { UploadSection } from './components/UploadSection';
import { CobeGlobe } from './components/CobeGlobe';
import { ModelRadarChart } from './components/ModelRadarChart';
import { AnimatedTooltip } from './components/AnimatedTooltip';
import { Boxes } from './components/ui/background-boxes';
import { NoiseBackground } from './components/ui/noise-background';
import { verificationAPI } from './services/api';
import { VerificationResponse } from './types';
import firstAvatarImage from '../DOCUMENTATION/photos/dc3a2408-258e-41a3-86c5-a4a7db781463.jpeg';
import secondAvatarImage from '../DOCUMENTATION/photos/WhatsApp Image 2026-04-09 at 08.10.00.jpeg';
import thirdAvatarImage from '../DOCUMENTATION/photos/1765992088545.jpeg';
import fourthAvatarImage from '../DOCUMENTATION/photos/1760198792329.jpeg';

const ANALYSIS_STEPS = [
      'UNDERGOING CNN ANALYSIS',
      'UNDERGOING VIT ANALYSIS',
      'UNDERGOING GAN ANALYSIS',
      'UNDERGOING OCR ANALYSIS',
];

const LANDING_TOOLTIP_ITEMS = [
      {
            id: 1,
            name: 'Jhashank Nayan',
            designation: 'Convolutional Feature Scan',
            image: firstAvatarImage,
      },
      {
            id: 2,
            name: 'Abhishek Singh',
            designation: 'Vision Transformer Mapping',
            image: secondAvatarImage,
      },
      {
            id: 3,
            name: 'Ayush Kaushik',
            designation: 'Synthetic Artifact Detection',
            image: thirdAvatarImage,
      },
      {
            id: 4,
            name: 'Hardik Singh',
            designation: 'Metadata and Text Integrity',
            image: fourthAvatarImage,
      },
] as const;

interface AnalysisHistory {
      result: VerificationResponse;
      timestamp: Date;
      fileName: string;
}

function App() {
      const [showLanding, setShowLanding] = useState(true);
      const [uploadedFile, setUploadedFile] = useState<File | null>(null);
      const [preview, setPreview] = useState<string | null>(null);
      const [uploadProgress, setUploadProgress] = useState(0);
      const [isAnalyzing, setIsAnalyzing] = useState(false);
      const [result, setResult] = useState<VerificationResponse | null>(null);
      const [error, setError] = useState<string | null>(null);
      const [showResultReady, setShowResultReady] = useState(false);
      const [activeAnalysisStep, setActiveAnalysisStep] = useState(0);
      const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistory[]>([]);
      const resultReadyTimeoutRef = useRef<number | null>(null);

      const handleFileSelect = async (file: File, preview: string) => {
            setUploadedFile(file);
            setPreview(preview);
            setError(null);
            setResult(null);
      };

      const handleRemoveImage = () => {
            setUploadedFile(null);
            setPreview(null);
            setUploadProgress(0);
            setResult(null);
            setError(null);
      };

      const handleAnalyze = async () => {
            if (!uploadedFile) return;

            setIsAnalyzing(true);
            setError(null);
            setResult(null);
            setUploadProgress(0);

            // Generate random delay between 10-15 seconds
            const randomDelay = Math.random() * (15000 - 10000) + 10000;
            const startTime = Date.now();

            try {
                  const response = await verificationAPI.verifyImage(
                        uploadedFile,
                        undefined,
                        undefined,
                        undefined,
                        (progress) => setUploadProgress(progress)
                  );

                  // Wait for minimum loading duration
                  const elapsedTime = Date.now() - startTime;
                  const remainingDelay = randomDelay - elapsedTime;

                  if (remainingDelay > 0) {
                        await new Promise((resolve) => setTimeout(resolve, remainingDelay));
                  }

                  setResult(response);
                  setAnalysisHistory((prev) =>
                        [
                              {
                                    result: response,
                                    timestamp: new Date(),
                                    fileName: uploadedFile.name,
                              },
                              ...prev,
                        ].slice(0, 2)
                  );
            } catch (err) {
                  const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
                  setError(errorMessage);
            } finally {
                  setIsAnalyzing(false);
                  setUploadProgress(0);
            }
      };

      const handleRetry = () => {
            handleRemoveImage();
      };

      useEffect(() => {
            if (result && !isAnalyzing) {
                  setShowResultReady(true);

                  if (resultReadyTimeoutRef.current) {
                        window.clearTimeout(resultReadyTimeoutRef.current);
                  }

                  resultReadyTimeoutRef.current = window.setTimeout(() => {
                        setShowResultReady(false);
                  }, 400);
            } else {
                  setShowResultReady(false);
            }
      }, [result, isAnalyzing]);

      useEffect(() => {
            return () => {
                  if (resultReadyTimeoutRef.current) {
                        window.clearTimeout(resultReadyTimeoutRef.current);
                  }
            };
      }, []);

      useEffect(() => {
            if (!isAnalyzing) {
                  setActiveAnalysisStep(0);
                  return;
            }

            const stepInterval = window.setInterval(() => {
                  setActiveAnalysisStep((prevStep) => (prevStep + 1) % ANALYSIS_STEPS.length);
            }, 1300);

            return () => {
                  window.clearInterval(stepInterval);
            };
      }, [isAnalyzing]);

      if (showLanding) {
            return (
                  <div className="relative h-screen overflow-hidden bg-[#000000] flex flex-col">
                        <div className="absolute inset-0 pointer-events-none overflow-hidden">
                              <Boxes className="opacity-95 mix-blend-screen [mask-image:radial-gradient(ellipse_at_center,black_78%,transparent_100%)]" />
                              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,0,0,0.03),rgba(0,0,0,0.52)_72%)]" />
                        </div>
                        <div className="absolute -top-24 -left-24 w-[420px] h-[420px] rounded-full bg-[#00E5FF]/15 blur-3xl pointer-events-none" />
                        <div className="absolute -bottom-28 -right-28 w-[460px] h-[460px] rounded-full bg-[#7B61FF]/18 blur-3xl pointer-events-none" />
                        <div className="absolute top-[28%] left-[46%] w-[260px] h-[260px] rounded-full bg-[#E91E8C]/12 blur-3xl pointer-events-none" />

                        <div className="absolute bottom-14 left-1/2 z-20 hidden -translate-x-1/2 sm:flex items-center gap-3">
                              <p className="text-[11px] tracking-[0.2em] uppercase text-[#E7DEFF]/75 whitespace-nowrap">Developed by</p>
                              <AnimatedTooltip items={[...LANDING_TOOLTIP_ITEMS]} />
                        </div>

                        <main className="flex-1 relative z-10 flex items-center px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
                              <div className="w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-6 items-center">
                                    <section className="lg:col-span-7 text-center lg:text-left max-w-3xl mx-auto lg:mx-0">
                                          <p className="text-xs font-semibold tracking-[0.24em] uppercase text-[#00E5FF]">VeriSight Intelligence Platform</p>
                                          <h1 className="mt-3 text-4xl sm:text-5xl xl:text-6xl font-black leading-[1.02] text-[#F3ECFF]">
                                                AI-POWERED VISUAL
                                                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-[#00E5FF] via-[#F3ECFF] to-[#7B61FF]">
                                                      VERIFICATION SYSTEM
                                                </span>
                                          </h1>
                                          <p className="mt-5 max-w-2xl mx-auto lg:mx-0 text-sm md:text-base text-[#E7DEFF]/80 leading-relaxed">
                                                For delivery platform fraud detection with CNN, ViT, GAN, and OCR fusion for robust decision confidence.
                                          </p>

                                          <div className="mt-7 flex flex-wrap gap-3 text-xs justify-center lg:justify-start">
                                                <span className="px-3 py-1.5 rounded-full border border-[#00E5FF]/40 bg-[#00E5FF]/10 text-[#00E5FF]">4 AI LAYERS</span>
                                                <span className="px-3 py-1.5 rounded-full border border-[#7B61FF]/40 bg-[#7B61FF]/10 text-[#F3ECFF]">10-15 SEC TYPICAL</span>
                                                <span className="px-3 py-1.5 rounded-full border border-[#E91E8C]/40 bg-[#E91E8C]/10 text-[#F3ECFF]">DECISION + CONFIDENCE</span>
                                          </div>

                                          <div className="mt-10 flex justify-center lg:justify-start">
                                                <NoiseBackground
                                                      containerClassName="w-fit p-2 rounded-full"
                                                      gradientColors={['rgb(255, 100, 150)', 'rgb(100, 150, 255)', 'rgb(255, 200, 100)']}
                                                >
                                                      <button
                                                            onClick={() => setShowLanding(false)}
                                                            className="h-full w-full cursor-pointer rounded-full bg-gradient-to-r from-black via-black to-neutral-900 px-8 py-3 text-white text-sm md:text-base font-semibold tracking-[0.12em] uppercase shadow-[0px_1px_0px_0px_var(--color-neutral-950)_inset,0px_1px_0px_0px_var(--color-neutral-800)] transition-all duration-100 active:scale-95"
                                                      >
                                                            GET STARTED →
                                                      </button>
                                                </NoiseBackground>
                                          </div>
                                    </section>

                                    <section className="lg:col-span-5 relative h-[380px] sm:h-[460px] lg:h-[540px] mx-auto w-full max-w-[520px]">
                                          <div className="h-full flex flex-col items-center">
                                                <p className="pt-3 sm:pt-8 text-center text-sm sm:text-xl font-black tracking-[0.12em] sm:tracking-[0.16em] uppercase text-transparent bg-clip-text bg-gradient-to-r from-[#00E5FF] via-[#F3ECFF] to-[#7B61FF] drop-shadow-[0_0_12px_rgba(0,229,255,0.28)] whitespace-nowrap">
                                                      DEEPFAKE IMAGE FORENSICS
                                                </p>

                                                <div className="relative flex-1 w-full flex items-center justify-center py-4 sm:py-6">
                                                      <div className="absolute w-[300px] h-[300px] sm:w-[360px] sm:h-[360px] rounded-full border border-[#7B61FF]/30" />
                                                      <div className="absolute w-[265px] h-[265px] sm:w-[320px] sm:h-[320px] rounded-full border border-dashed border-[#00E5FF]/35 animate-spin" style={{ animationDuration: '20s' }} />
                                                      <div className="absolute w-[230px] h-[230px] sm:w-[285px] sm:h-[285px] rounded-full border border-[#E91E8C]/25 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '26s' }} />

                                                      <div className="relative w-[205px] h-[205px] sm:w-[255px] sm:h-[255px] rounded-full border border-[#00E5FF]/35 bg-[#020204]/65 backdrop-blur-sm shadow-[0_0_35px_rgba(0,229,255,0.2)] p-3">
                                                            <CobeGlobe
                                                                  className="w-full h-full"
                                                                  dark={1}
                                                                  scale={1.08}
                                                                  diffuse={1.25}
                                                                  baseColor="#2A2D6E"
                                                                  markerColor="#00E5FF"
                                                                  glowColor="#7B61FF"
                                                                  autoRotateSpeed={0.006}
                                                                  draggable={true}
                                                                  ariaLabel="Landing preview globe"
                                                            />
                                                      </div>
                                                </div>

                                                <p className="pb-4 text-[11px] tracking-[0.24em] text-[#E7DEFF]/70 uppercase text-center whitespace-nowrap">
                                                      CNN • VIT • GAN • OCR
                                                </p>
                                          </div>
                                    </section>
                              </div>
                        </main>

                        <footer className="relative z-10 border-t border-[#7B61FF]/25 bg-[#000000]/80 backdrop-blur-sm">
                              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 text-center">
                                    <p className="text-[11px] sm:text-xs text-[#E7DEFF]/70">© 2026 VeriSight. All rights reserved.</p>
                              </div>
                        </footer>
                  </div>
            );
      }

      const layerKeys: Array<keyof VerificationResponse['layer_scores']> = ['cnn', 'vit', 'gan', 'ocr'];

      const isLayerAvailable = (status: string): boolean => {
            const normalizedStatus = status.toLowerCase();
            return normalizedStatus === 'loaded' || normalizedStatus === 'ok' || normalizedStatus === 'active' || normalizedStatus === 'ready';
      };

      return (
            <div className="relative min-h-screen bg-[#000000] flex flex-col">
                  <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                              backgroundImage:
                                    'radial-gradient(rgba(0,229,255,0.15) 0.8px, transparent 0.8px), radial-gradient(rgba(233,30,140,0.1) 0.8px, transparent 0.8px)',
                              backgroundSize: '22px 22px, 34px 34px',
                              backgroundPosition: '0 0, 11px 11px',
                              opacity: 0.35,
                        }}
                  />
                  <Header />

                  <main className="flex-1 relative z-10 overflow-y-auto">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
                              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                                    <section className="bg-[rgba(0,0,0,0.78)] rounded-2xl border border-[#7B61FF]/35 shadow-[0_0_20px_rgba(123,97,255,0.2)] p-4 sm:p-5 flex flex-col min-h-fit lg:min-h-[500px]">
                                          <div className="mb-3 sm:mb-4">
                                                <p className="text-xs font-semibold tracking-wide uppercase text-[#00E5FF]">Upload</p>
                                                <h2 className="text-xl sm:text-2xl font-bold text-[#F3ECFF] mt-1">Input Image</h2>
                                                <p className="text-sm text-[#E7DEFF]/75 mt-1">
                                                      Upload one image, then click Analyze Image.
                                                </p>
                                          </div>

                                          {!preview ? (
                                                <div className="flex-1 flex items-center justify-center">
                                                      <UploadSection
                                                            onFileSelect={handleFileSelect}
                                                            isLoading={isAnalyzing}
                                                            uploadProgress={uploadProgress}
                                                      />
                                                </div>
                                          ) : (
                                                <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                                                      <div className="border border-[#7B61FF]/30 rounded-xl overflow-hidden bg-[#050505]">
                                                            <div className="relative">
                                                                  <img src={preview} alt="Selected" className="w-full h-56 object-contain bg-[#050505]" />
                                                                  {isAnalyzing && (
                                                                        <div className="absolute inset-0 pointer-events-none overflow-hidden">
                                                                              <div className="scan-overlay" />
                                                                              <div className="scan-line" />
                                                                              <div className="absolute inset-x-0 bottom-2 flex justify-center">
                                                                                    <span className="px-3 py-1 rounded-full border border-[#00E5FF]/45 bg-[#050510]/80 text-[10px] font-semibold tracking-[0.18em] text-[#00E5FF]">
                                                                                          SCANNING IMAGE
                                                                                    </span>
                                                                              </div>
                                                                        </div>
                                                                  )}
                                                            </div>
                                                            <div className="p-3 flex items-center justify-between">
                                                                  <p className="text-xs text-[#F3ECFF]/80 truncate">{uploadedFile?.name}</p>
                                                                  {!isAnalyzing && (
                                                                        <button
                                                                              onClick={handleRemoveImage}
                                                                              className="text-xs px-3 py-1 bg-[#0B0B0B] hover:bg-[#161616] rounded-md text-[#F3ECFF] border border-[#7B61FF]/35"
                                                                        >
                                                                              Remove
                                                                        </button>
                                                                  )}
                                                            </div>
                                                      </div>

                                                      {!isAnalyzing && (
                                                            <button
                                                                  onClick={handleAnalyze}
                                                                  className="w-full py-3 rounded-lg transition-all duration-200 active:scale-95 border border-[#7B61FF]/45 bg-[rgba(42,45,110,0.5)] text-[#F3ECFF] font-semibold uppercase tracking-wide hover:bg-[rgba(42,45,110,0.65)] hover:border-[#9C7BFF]"
                                                            >
                                                                  ANALYZE IMAGE
                                                            </button>
                                                      )}

                                                      <p className="text-xs text-[#E7DEFF]/65">Accepted formats: JPEG, PNG, WebP, BMP · Max 10MB</p>
                                                </div>
                                          )}
                                    </section>

                                    <section className="bg-[rgba(0,0,0,0.78)] rounded-2xl border border-[#7B61FF]/35 shadow-[0_0_20px_rgba(123,97,255,0.2)] p-4 sm:p-5 flex flex-col min-h-fit lg:min-h-[500px]">
                                          <p className="text-xs font-semibold tracking-wide uppercase text-[#00E5FF]">Analysis</p>
                                          <h2 className="text-xl sm:text-2xl font-bold text-[#F3ECFF] mt-1">Verification Output</h2>

                                          {!preview && (
                                                <div className="mt-4 border border-dashed border-[#7B61FF]/30 rounded-xl flex items-center justify-center bg-[#050505] p-4 min-h-[300px] sm:min-h-[400px]">
                                                      <CobeGlobe
                                                            className="w-[360px] h-[360px]"
                                                            dark={1}
                                                            scale={1.02}
                                                            diffuse={1.25}
                                                            baseColor="#2A2D6E"
                                                            markerColor="#00E5FF"
                                                            glowColor="#7B61FF"
                                                            autoRotateSpeed={0.004}
                                                            draggable={true}
                                                      />
                                                </div>
                                          )}

                                          {preview && isAnalyzing && (
                                                <div className="mt-4 rounded-xl border border-[#7B61FF]/30 bg-[#050505] p-4 max-h-[500px] overflow-y-auto">
                                                      <p className="text-xs font-semibold tracking-[0.18em] text-[#00E5FF]">ANALYSIS IN PROGRESS</p>
                                                      <div className="mt-4 space-y-2">
                                                            {ANALYSIS_STEPS.map((step, index) => {
                                                                  const isDone = index < activeAnalysisStep;
                                                                  const isActive = index === activeAnalysisStep;

                                                                  return (
                                                                        <div
                                                                              key={step}
                                                                              className={`rounded-lg border px-3 py-2 text-xs font-semibold tracking-wide transition-all ${isActive
                                                                                    ? 'border-[#00E5FF]/55 bg-[#00E5FF]/10 text-[#00E5FF] shadow-[0_0_14px_rgba(0,229,255,0.14)]'
                                                                                    : isDone
                                                                                          ? 'border-[#7B61FF]/45 bg-[#7B61FF]/12 text-[#F3ECFF]'
                                                                                          : 'border-[#7B61FF]/25 bg-[#0B0B0B] text-[#E7DEFF]/65'
                                                                                    }`}
                                                                        >
                                                                              <div className="flex items-center justify-between">
                                                                                    <span>{step}</span>
                                                                                    <span className="text-[10px]">
                                                                                          {isActive ? 'RUNNING' : isDone ? 'DONE' : 'PENDING'}
                                                                                    </span>
                                                                              </div>
                                                                        </div>
                                                                  );
                                                            })}
                                                      </div>
                                                      <p className="text-xs text-[#E7DEFF]/65 mt-4">Please wait while models process your image.</p>
                                                </div>
                                          )}

                                          {preview && error && !isAnalyzing && (
                                                <div className="mt-4 p-4 rounded-xl border border-[#E91E8C]/50 bg-[#1A1A34]">
                                                      <p className="text-sm font-semibold text-[#F3ECFF]">Analysis failed</p>
                                                      <p className="text-sm text-[#E7DEFF]/80 mt-1">{error}</p>
                                                      <button
                                                            onClick={handleRetry}
                                                            className="mt-3 px-4 py-2 text-white text-sm rounded-lg hover:brightness-110 border border-[#00E5FF]/60"
                                                            style={{ background: 'linear-gradient(135deg, #00e5ff, #7b61ff)' }}
                                                      >
                                                            Try Again
                                                      </button>
                                                </div>
                                          )}

                                          {preview && result && !isAnalyzing && (
                                                <div className="mt-4 overflow-y-auto flex flex-col gap-4 max-h-[600px]">
                                                      {showResultReady ? (
                                                            <div className="h-full rounded-xl border border-[#7B61FF]/35 bg-[#050505] flex items-center justify-center">
                                                                  <div className="text-center result-ready-anim result-ready-glow px-6 py-4 rounded-xl border border-[#00E5FF]/25 bg-[#080813]/75">
                                                                        <p className="text-[#00E5FF] text-[11px] tracking-[0.26em] font-semibold">RESULT</p>
                                                                        <p className="text-[#F3ECFF] text-3xl md:text-4xl font-bold mt-2">READY</p>
                                                                        <p className="text-[#E7DEFF]/70 text-[11px] mt-3 tracking-[0.14em]">DISPLAYING MODEL INSIGHTS</p>
                                                                  </div>
                                                            </div>
                                                      ) : (
                                                            <>
                                                                  <div
                                                                        className="rounded-xl p-3 border sticky top-0 z-10 bg-[#070707]"
                                                                        style={{
                                                                              backgroundColor: '#070707',
                                                                              borderColor: '#7B61FF66',
                                                                        }}
                                                                  >
                                                                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
                                                                              <div>
                                                                                    <p className="text-[11px] font-semibold tracking-wide uppercase text-[#E7DEFF]/65">Final Decision</p>
                                                                                    <p className="text-lg sm:text-xl font-bold mt-1" style={{ color: result.abstained ? '#F3ECFF' : '#00E5FF' }}>
                                                                                          {result.abstained ? 'INCONCLUSIVE' : result.decision}
                                                                                    </p>
                                                                              </div>
                                                                              <div className="text-right">
                                                                                    <p className="text-[11px] text-[#E7DEFF]/65">Confidence</p>
                                                                                    <p className="text-sm font-semibold text-[#F3ECFF]">
                                                                                          {Math.round(result.confidence * 100)}%
                                                                                    </p>
                                                                              </div>
                                                                        </div>
                                                                  </div>

                                                                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                                                        <div className="rounded-xl border border-[#7B61FF]/30 bg-[#050505] p-4">
                                                                              <p className="text-sm font-semibold text-[#F3ECFF] mb-3">Model Readings</p>
                                                                              <div className="grid grid-cols-2 sm:grid-cols-2 gap-3">
                                                                                    {layerKeys.map((layer) => {
                                                                                          const score = result.layer_scores[layer];
                                                                                          const reliability = result.layer_reliabilities[layer];
                                                                                          const status = result.layer_status[layer];
                                                                                          const layerReady = isLayerAvailable(status);
                                                                                          return (
                                                                                                <div key={layer} className="rounded-lg border border-[#7B61FF]/25 bg-[#0B0B0B] p-3">
                                                                                                      <div className="flex items-center justify-between">
                                                                                                            <p className="uppercase text-xs font-bold tracking-wide text-[#00E5FF]">{layer}</p>
                                                                                                            <span className={`text-[10px] px-2 py-0.5 rounded-full ${status === 'loaded' ? 'bg-[#00E5FF]/20 text-[#00E5FF]' : 'bg-[#050505] text-[#E7DEFF]/65'}`}>
                                                                                                                  {status}
                                                                                                            </span>
                                                                                                      </div>
                                                                                                      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                                                                                                            <div>
                                                                                                                  <p className="text-[#E7DEFF]/65">Score</p>
                                                                                                                  <p className="text-[#F3ECFF] font-semibold">{layerReady ? score.toFixed(1) : '--'}</p>
                                                                                                            </div>
                                                                                                            <div>
                                                                                                                  <p className="text-[#E7DEFF]/65">Reliability</p>
                                                                                                                  <p className="text-[#F3ECFF] font-semibold">{layerReady ? `${Math.round(reliability * 100)}%` : '--'}</p>
                                                                                                            </div>
                                                                                                      </div>
                                                                                                </div>
                                                                                          );
                                                                                    })}
                                                                              </div>
                                                                        </div>

                                                                        <div className="rounded-xl border border-[#7B61FF]/30 bg-[#050505] p-4">
                                                                              <p className="text-sm font-semibold text-[#F3ECFF]">Radar Chart</p>
                                                                              <p className="text-[11px] text-[#E7DEFF]/65 mt-1">All 4 models: score vs reliability</p>
                                                                              <div className="mt-2 rounded-lg border border-[#7B61FF]/25 bg-[#0B0B0B] p-2 h-[300px] sm:h-[350px]">
                                                                                    <ModelRadarChart result={result} />
                                                                              </div>
                                                                        </div>
                                                                  </div>

                                                                  <button
                                                                        onClick={handleRetry}
                                                                        className="w-full py-2.5 rounded-lg font-semibold uppercase tracking-wide transition-colors border border-[#7B61FF]/45 bg-[rgba(42,45,110,0.5)] text-[#F3ECFF] hover:bg-[rgba(42,45,110,0.65)] hover:border-[#9C7BFF]"
                                                                  >
                                                                        ANALYZE ANOTHER IMAGE
                                                                  </button>
                                                            </>
                                                      )}
                                                </div>
                                          )}
                                    </section>
                              </div>
                        </div>
                  </main>

                  {analysisHistory.length > 0 && (
                        <section className="relative z-10 border-t border-[#7B61FF]/25 bg-[#000000]/80 backdrop-blur-sm px-4 sm:px-6 lg:px-8 py-4 sm:py-5">
                              <div className="max-w-7xl mx-auto">
                                    <p className="text-xs font-semibold tracking-wide uppercase text-[#00E5FF] mb-3 sm:mb-4">Recent Analysis</p>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                                          {analysisHistory.map((history, index) => {
                                                const timeAgo = Math.floor((Date.now() - history.timestamp.getTime()) / 1000);
                                                const timeStr = timeAgo < 60 ? `${timeAgo}s ago` : `${Math.floor(timeAgo / 60)}m ago`;

                                                return (
                                                      <div
                                                            key={index}
                                                            className="bg-[rgba(0,0,0,0.5)] rounded-lg border border-[#7B61FF]/30 p-4 hover:border-[#7B61FF]/50 transition-colors"
                                                      >
                                                            <div className="flex items-start justify-between mb-3">
                                                                  <div className="flex-1 min-w-0">
                                                                        <p className="text-xs text-[#E7DEFF]/65 truncate">{history.fileName}</p>
                                                                        <p className="text-[10px] text-[#E7DEFF]/50 mt-1">{timeStr}</p>
                                                                  </div>
                                                                  <div className="text-right ml-3">
                                                                        <p className="text-xs font-semibold tracking-wide uppercase" style={{ color: history.result.abstained ? '#F3ECFF' : '#00E5FF' }}>
                                                                              {history.result.abstained ? 'INCONCLUSIVE' : history.result.decision}
                                                                        </p>
                                                                        <p className="text-xs text-[#F3ECFF] font-semibold mt-1">
                                                                              {Math.round(history.result.confidence * 100)}%
                                                                        </p>
                                                                  </div>
                                                            </div>
                                                            <div className="flex gap-2 flex-wrap">
                                                                  {(['cnn', 'vit', 'gan', 'ocr'] as const).map((layer) => {
                                                                        const status = history.result.layer_status[layer];
                                                                        const normalizedStatus = status.toLowerCase();
                                                                        const isAvailable = normalizedStatus === 'loaded' || normalizedStatus === 'ok' || normalizedStatus === 'active' || normalizedStatus === 'ready';

                                                                        return (
                                                                              <span
                                                                                    key={layer}
                                                                                    className={`text-[10px] font-bold tracking-wide px-2 py-1 rounded-full ${isAvailable
                                                                                          ? 'bg-[#00E5FF]/15 text-[#00E5FF] border border-[#00E5FF]/30'
                                                                                          : 'bg-[#7B61FF]/15 text-[#7B61FF] border border-[#7B61FF]/30'
                                                                                    }`}
                                                                              >
                                                                                    {layer.toUpperCase()}
                                                                              </span>
                                                                        );
                                                                  })}
                                                            </div>
                                                      </div>
                                                );
                                          })}
                                    </div>
                              </div>
                        </section>
                  )}

                  <footer className="relative z-10 border-t border-[#7B61FF]/25 bg-[#000000]/80 backdrop-blur-sm px-4 sm:px-6 lg:px-8 py-3 sm:py-4 mt-auto">
                        <div className="max-w-7xl mx-auto text-center">
                              <p className="text-[10px] sm:text-xs text-[#E7DEFF]/70">© 2026 VeriSight. All rights reserved.</p>
                        </div>
                  </footer>
            </div>
      );
}

export default App;
