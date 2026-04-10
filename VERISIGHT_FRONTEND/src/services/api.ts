import axios, { AxiosInstance } from 'axios';
import { API_CONFIG } from '../config';
import { VerificationResponse } from '../types';

class VerificationAPI {
      private client: AxiosInstance;

      constructor() {
            this.client = axios.create({
                  baseURL: API_CONFIG.BASE_URL,
                  timeout: API_CONFIG.TIMEOUT,
                  headers: {
                        'Accept': 'application/json',
                  },
            });
      }

      async verifyImage(
            file: File,
            orderDate?: string,
            deliveryDate?: string,
            mfgDateClaimed?: string,
            onUploadProgress?: (progress: number) => void
      ): Promise<VerificationResponse> {
            const formData = new FormData();
            formData.append('image', file);

            if (orderDate) formData.append('order_date', orderDate);
            if (deliveryDate) formData.append('delivery_date', deliveryDate);
            if (mfgDateClaimed) formData.append('mfg_date_claimed', mfgDateClaimed);

            try {
                  const response = await this.client.post<VerificationResponse>(
                        API_CONFIG.VERIFY_ENDPOINT,
                        formData,
                        {
                              onUploadProgress: (progressEvent) => {
                                    if (progressEvent.total) {
                                          const percentCompleted = Math.round(
                                                (progressEvent.loaded * 100) / progressEvent.total
                                          );
                                          onUploadProgress?.(percentCompleted);
                                    }
                              },
                        }
                  );

                  return response.data;
            } catch (error) {
                  if (axios.isAxiosError(error)) {
                        if (error.response?.status === 400) {
                              throw new Error('Invalid file format or missing required field');
                        } else if (error.response?.status === 413) {
                              throw new Error('File too large. Maximum size is 10MB');
                        } else if (error.response?.status === 422) {
                              throw new Error('Could not process image. Please ensure it is a valid image file');
                        } else if (error.response?.status === 503) {
                              throw new Error('System is warming up. Please try again in a moment');
                        } else if (error.response?.data?.error) {
                              throw new Error(error.response.data.error);
                        } else if (error.response?.data?.detail) {
                              throw new Error(error.response.data.detail);
                        } else if (error.code === 'ECONNABORTED') {
                              throw new Error('Request timeout. Please check your connection and try again');
                        }
                  }
                  throw new Error('Failed to verify image. Please try again');
            }
      }

      checkHealth(): Promise<boolean> {
            const healthUrl = new URL(API_CONFIG.HEALTH_ENDPOINT, API_CONFIG.BASE_URL).toString();

            return this.client
                  .get(healthUrl)
                  .then(() => true)
                  .catch(() => false);
      }
}

export const verificationAPI = new VerificationAPI();
