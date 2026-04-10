export function LoadingSkeleton() {
      return (
            <div className="w-full max-w-4xl mx-auto space-y-6 mt-8">
                  <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6 animate-pulse">
                        <div className="h-12 bg-slate-200 rounded-lg w-1/3 mb-4"></div>
                        <div className="h-24 bg-slate-100 rounded-lg"></div>
                  </div>

                  <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6">
                        <div className="h-8 bg-slate-200 rounded-lg w-1/4 mb-6 animate-pulse"></div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {[1, 2, 3, 4].map((i) => (
                                    <div key={i} className="border border-slate-200 rounded-lg p-4 animate-pulse">
                                          <div className="flex justify-between items-start mb-3">
                                                <div>
                                                      <div className="h-5 bg-slate-200 rounded w-16 mb-2"></div>
                                                      <div className="h-4 bg-slate-100 rounded w-32"></div>
                                                </div>
                                                <div className="h-6 bg-slate-200 rounded w-16"></div>
                                          </div>

                                          <div className="space-y-3">
                                                <div className="h-4 bg-slate-100 rounded w-1/3 mb-2"></div>
                                                <div className="h-2 bg-slate-200 rounded-full"></div>

                                                <div className="grid grid-cols-2 gap-2">
                                                      <div className="h-12 bg-slate-100 rounded"></div>
                                                      <div className="h-12 bg-slate-100 rounded"></div>
                                                </div>

                                                <div className="h-3 bg-slate-100 rounded w-full"></div>
                                          </div>
                                    </div>
                              ))}
                        </div>
                  </div>

                  <div className="flex justify-center">
                        <div className="text-slate-600 text-center">
                              <p className="font-medium">Analyzing image...</p>
                              <p className="text-sm text-slate-500 mt-2">Please wait, this typically takes 1-3 seconds</p>
                        </div>
                  </div>
            </div>
      );
}
