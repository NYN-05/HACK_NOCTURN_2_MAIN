interface ErrorMessageProps {
      message: string;
      onRetry?: () => void;
}

export function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
      return (
            <div className="w-full max-w-2xl mx-auto mt-8">
                  <div className="bg-red-50 border-2 border-red-200 rounded-xl p-6">
                        <div className="flex items-start gap-4">
                              <div className="text-2xl">!</div>
                              <div className="flex-1">
                                    <h4 className="text-lg font-bold text-red-900 mb-2">Verification Failed</h4>
                                    <p className="text-red-700 text-sm mb-4">{message}</p>

                                    <div className="space-y-2 text-sm text-red-600">
                                          <p>Possible solutions:</p>
                                          <ul className="list-disc list-inside space-y-1">
                                                <li>Ensure the image file is not corrupted</li>
                                                <li>Check that the file size is under 10MB</li>
                                                <li>Verify your internet connection</li>
                                                <li>Try using a different image format (JPEG, PNG, WebP, BMP)</li>
                                          </ul>
                                    </div>

                                    {onRetry && (
                                          <button
                                                onClick={onRetry}
                                                className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
                                          >
                                                Try Again
                                          </button>
                                    )}
                              </div>
                        </div>
                  </div>
            </div>
      );
}
