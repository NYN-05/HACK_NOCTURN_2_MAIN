import { FILE_CONFIG } from '../config';

export function formatFileSize(bytes: number): string {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

export function validateFile(file: File): { valid: boolean; error?: string } {
      if (!FILE_CONFIG.ACCEPTED_TYPES.includes(file.type)) {
            return {
                  valid: false,
                  error: `Unsupported file type. Accepted formats: ${FILE_CONFIG.ACCEPTED_EXTENSIONS.join(', ')}`,
            };
      }

      if (file.size > FILE_CONFIG.MAX_SIZE) {
            return {
                  valid: false,
                  error: `File too large. Maximum size is ${formatFileSize(FILE_CONFIG.MAX_SIZE)}`,
            };
      }

      return { valid: true };
}

export function formatConfidence(confidence: number): number {
      return Math.round(confidence * 100);
}

export function formatProcessingTime(ms: number): string {
      if (ms < 1000) {
            return `${ms}ms`;
      }
      return `${(ms / 1000).toFixed(2)}s`;
}

export function generatePreview(file: File): Promise<string> {
      return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                  const result = e.target?.result;
                  if (typeof result === 'string') {
                        resolve(result);
                  } else {
                        reject(new Error('Failed to generate preview'));
                  }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
      });
}

export function formatDate(date: string): string {
      try {
            return new Date(date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric',
            });
      } catch {
            return date;
      }
}
