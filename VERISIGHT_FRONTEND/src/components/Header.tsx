import { CobeGlobe } from './CobeGlobe';

export function Header() {
      return (
            <header className="relative overflow-hidden border-b border-[#7B61FF]/30 bg-[#000000]/95 backdrop-blur-md">
                  <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                              backgroundImage: 'radial-gradient(rgba(0,229,255,0.14) 0.7px, transparent 0.7px), radial-gradient(rgba(123,97,255,0.12) 0.7px, transparent 0.7px)',
                              backgroundSize: '18px 18px, 28px 28px',
                              backgroundPosition: '0 0, 9px 9px',
                              opacity: 0.45,
                        }}
                  />
                  <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                        <div className="flex items-center justify-between">
                              <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-full overflow-hidden border border-[#00E5FF]/60 shadow-[0_0_16px_rgba(0,229,255,0.28)] bg-[#0A0A14]">
                                          <CobeGlobe
                                                className="w-full h-full"
                                                theta={0.2}
                                                dark={1}
                                                scale={1.02}
                                                diffuse={1.2}
                                                mapSamples={20000}
                                                mapBrightness={6}
                                                baseColor="#2A2D6E"
                                                markerColor="#00E5FF"
                                                glowColor="#7B61FF"
                                                autoRotateSpeed={0.006}
                                                draggable={false}
                                                ariaLabel="VeriSight Globe"
                                          />
                                    </div>
                                    <div>
                                          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">VeriSight</h1>
                                          <p className="text-[#E7DEFF]/80 text-xs md:text-sm">Image Authenticity Intelligence</p>
                                    </div>
                              </div>

                              <div className="hidden md:flex items-center gap-2">
                                    <div className="px-3 py-1.5 rounded-full border border-[#7B61FF]/40 bg-[#090909] text-xs font-medium text-[#F3ECFF]">
                                          v2.0.0
                                    </div>
                                    <div className="px-3 py-1.5 rounded-full border border-[#00E5FF]/55 bg-[#00E5FF]/10 text-xs font-medium text-[#00E5FF] shadow-[0_0_12px_rgba(0,229,255,0.35)]">
                                          Secure AI Verification
                                    </div>
                              </div>
                        </div>
                  </div>
            </header>
      );
}
