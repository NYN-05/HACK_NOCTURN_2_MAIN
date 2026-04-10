import { useRef, useState } from 'react';
import { generatePreview, validateFile } from '../utils/helpers';
import { DottedGlowBackground } from './ui/dotted-glow-background';

interface UploadSectionProps {
      onFileSelect: (file: File, preview: string) => void;
      isLoading: boolean;
      uploadProgress: number;
}

export function UploadSection({ onFileSelect, isLoading, uploadProgress }: UploadSectionProps) {
      const inputRef = useRef<HTMLInputElement>(null);
      const [dragActive, setDragActive] = useState(false);
      const [error, setError] = useState<string | null>(null);

      const handleFile = async (file: File) => {
            setError(null);
            const validation = validateFile(file);

            if (!validation.valid) {
                  setError(validation.error!);
                  return;
            }

            try {
                  const preview = await generatePreview(file);
                  onFileSelect(file, preview);
            } catch (err) {
                  setError('Failed to process image. Please try again.');
            }
      };

      const handleDrag = (e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.type === 'dragenter' || e.type === 'dragover') {
                  setDragActive(true);
            } else if (e.type === 'dragleave') {
                  setDragActive(false);
            }
      };

      const handleDrop = (e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);

            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleFile(e.dataTransfer.files[0]);
            }
      };

      const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            if (e.target.files && e.target.files[0]) {
                  handleFile(e.target.files[0]);
            }
      };

      const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
            if (isLoading) return;
            if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  inputRef.current?.click();
            }
      };

      return (
            <div className="w-full max-w-2xl mx-auto">
                  <div
                        role="button"
                        tabIndex={0}
                        aria-label="Upload image for authenticity verification"
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                        onClick={() => !isLoading && inputRef.current?.click()}
                        onKeyDown={handleKeyDown}
                        className={`relative overflow-hidden border-2 border-dashed rounded-2xl px-8 py-12 text-center cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[#00E5FF] focus:ring-offset-2 focus:ring-offset-[#000000] ${dragActive
                              ? 'border-[#00E5FF] bg-[#0B0B0B] shadow-[0_0_20px_rgba(0,229,255,0.3)]'
                              : 'border-[#7B61FF]/35 bg-[#050505] hover:border-[#E91E8C] hover:shadow-[0_0_16px_rgba(233,30,140,0.25)]'
                              } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                        <DottedGlowBackground
                              className="pointer-events-none absolute inset-0 mask-radial-to-90% mask-radial-at-center opacity-10 dark:opacity-30"
                              opacity={0.45}
                              gap={10}
                              radius={1.6}
                              colorLightVar="--color-neutral-500"
                              glowColorLightVar="--color-neutral-600"
                              colorDarkVar="--color-neutral-500"
                              glowColorDarkVar="--color-sky-800"
                              backgroundOpacity={0}
                              speedMin={0.3}
                              speedMax={1.6}
                              speedScale={1}
                        />

                        <input
                              ref={inputRef}
                              type="file"
                              accept=".jpg,.jpeg,.png,.webp,.bmp"
                              onChange={handleInputChange}
                              disabled={isLoading}
                              title="Upload image file"
                              aria-label="Upload image file"
                              className="hidden"
                        />

                        {isLoading ? (
                              <div className="relative z-10 space-y-4">
                                    <div className="upload-loading-badge w-14 h-14 rounded-full flex items-center justify-center mx-auto animate-pulse text-white font-bold shadow-[0_0_20px_rgba(233,30,140,0.5),0_0_40px_rgba(0,229,255,0.35)]">
                                          <span>AI</span>
                                    </div>
                                    <div>
                                          <p className="text-lg font-semibold text-white">Uploading...</p>
                                          <p className="text-sm text-[#E7DEFF]/70 mt-2">{uploadProgress}% complete</p>
                                    </div>
                                    <progress
                                          className="upload-progress w-full"
                                          value={uploadProgress}
                                          max={100}
                                          aria-label="Upload progress"
                                    />
                              </div>
                        ) : (
                              <div className="relative z-10 space-y-3">
                                    <div className="w-14 h-14 bg-[#000000] text-[#00E5FF] rounded-full flex items-center justify-center mx-auto font-semibold text-sm border border-[#7B61FF]/35">
                                          FILE
                                    </div>
                                    <div>
                                          <p className="text-lg font-semibold text-white">Drop image here</p>
                                          <p className="text-sm text-[#E7DEFF]/75 mt-1">or click to browse from your device</p>
                                    </div>
                                    <p className="text-xs text-[#E7DEFF]/65">Accepted formats: JPEG, PNG, WebP, BMP · Max size: 10MB</p>
                              </div>
                        )}
                  </div>

                  {error && (
                        <div className="mt-4 p-4 bg-[#090909] border border-[#E91E8C]/45 rounded-lg">
                              <p className="text-[#F3ECFF] text-sm font-medium">{error}</p>
                        </div>
                  )}
            </div>
      );
}
