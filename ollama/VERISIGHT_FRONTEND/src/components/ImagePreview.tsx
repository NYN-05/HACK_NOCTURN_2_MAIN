interface ImagePreviewProps {
      src: string;
      fileName?: string;
      onRemove?: () => void;
}

export function ImagePreview({ src, fileName, onRemove }: ImagePreviewProps) {
      return (
            <div className="w-full max-w-2xl mx-auto mt-8">
                  <div className="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden">
                        <img src={src} alt="Preview" className="w-full h-96 object-cover" />

                        <div className="p-4 border-t border-slate-200">
                              <div className="flex justify-between items-center">
                                    <div>
                                          <p className="text-sm font-medium text-slate-700">Image Selected</p>
                                          {fileName && <p className="text-xs text-slate-500 mt-1 truncate">{fileName}</p>}
                                    </div>
                                    {onRemove && (
                                          <button
                                                onClick={onRemove}
                                                className="px-3 py-1 text-sm bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
                                          >
                                                Remove
                                          </button>
                                    )}
                              </div>
                        </div>
                  </div>
            </div>
      );
}
