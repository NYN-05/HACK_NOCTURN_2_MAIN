import { useState } from 'react';

interface FormFieldsProps {
      onSubmit: (data: { orderDate: string; deliveryDate: string; mfgDateClaimed: string }) => void;
      isLoading: boolean;
      previewImage: string | null;
}

export function FormFields({ onSubmit, isLoading, previewImage }: FormFieldsProps) {
      const [orderDate, setOrderDate] = useState('');
      const [deliveryDate, setDeliveryDate] = useState('');
      const [mfgDateClaimed, setMfgDateClaimed] = useState('');

      const handleSubmit = (e: React.FormEvent) => {
            e.preventDefault();
            onSubmit({
                  orderDate,
                  deliveryDate,
                  mfgDateClaimed,
            });
      };

      if (!previewImage) return null;

      return (
            <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto mt-8">
                  <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6 space-y-5">
                        <h2 className="text-lg font-bold text-slate-900">Additional Information</h2>
                        <p className="text-sm text-slate-600">Provide optional dates to enhance verification accuracy</p>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                              <div className="space-y-2">
                                    <label className="block text-sm font-medium text-slate-700">Order Date</label>
                                    <input
                                          type="date"
                                          value={orderDate}
                                          onChange={(e) => setOrderDate(e.target.value)}
                                          disabled={isLoading}
                                          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:cursor-not-allowed transition-colors"
                                          placeholder="mm/dd/yyyy"
                                    />
                                    <p className="text-xs text-slate-500">When the order was placed</p>
                              </div>

                              <div className="space-y-2">
                                    <label className="block text-sm font-medium text-slate-700">Delivery Date</label>
                                    <input
                                          type="date"
                                          value={deliveryDate}
                                          onChange={(e) => setDeliveryDate(e.target.value)}
                                          disabled={isLoading}
                                          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:cursor-not-allowed transition-colors"
                                          placeholder="mm/dd/yyyy"
                                    />
                                    <p className="text-xs text-slate-500">When the product arrived</p>
                              </div>

                              <div className="space-y-2">
                                    <label className="block text-sm font-medium text-slate-700">Manufacture Date</label>
                                    <input
                                          type="date"
                                          value={mfgDateClaimed}
                                          onChange={(e) => setMfgDateClaimed(e.target.value)}
                                          disabled={isLoading}
                                          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100 disabled:cursor-not-allowed transition-colors"
                                          placeholder="mm/dd/yyyy"
                                    />
                                    <p className="text-xs text-slate-500">Claimed manufacturing date</p>
                              </div>
                        </div>

                        <button
                              type="submit"
                              disabled={isLoading}
                              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-all duration-200 disabled:bg-slate-400 disabled:cursor-not-allowed active:scale-95"
                        >
                              {isLoading ? 'Analyzing Image...' : 'Verify Image'}
                        </button>
                  </div>
            </form>
      );
}
