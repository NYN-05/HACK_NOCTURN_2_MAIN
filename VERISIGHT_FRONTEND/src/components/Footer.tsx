export function Footer() {
      return (
            <footer className="bg-slate-900 text-slate-300 border-t border-slate-800 mt-16">
                  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                              <div>
                                    <h3 className="text-white font-bold mb-3">VeriSight</h3>
                                    <p className="text-sm">AI-powered image authenticity verification using a multi-layer detection pipeline.</p>
                              </div>

                              <div>
                                    <h4 className="text-white font-bold mb-3">Detection Layers</h4>
                                    <ul className="text-sm space-y-2">
                                          <li className="text-slate-400">CNN - Visual Artifact Detection</li>
                                          <li className="text-slate-400">ViT - Semantic Analysis</li>
                                          <li className="text-slate-400">GAN - Deepfake Detection</li>
                                          <li className="text-slate-400">OCR - Text Verification</li>
                                    </ul>
                              </div>
                        </div>

                        <div className="border-t border-slate-800 pt-8">
                              <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                                    <p className="text-sm text-slate-400">Copyright 2026 VeriSight. All rights reserved.</p>
                                    <p className="text-sm text-slate-400">API v2.0.0 · Typical processing time: 1-3 seconds · Max file size: 10MB</p>
                              </div>
                        </div>
                  </div>
            </footer>
      );
}
