// API Configuration
export const API_CONFIG = {
      BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
      VERIFY_ENDPOINT: '/verify',
      HEALTH_ENDPOINT: '/health',
      TIMEOUT: 30000, // 30 seconds
};

// File Upload Configuration
export const FILE_CONFIG = {
      MAX_SIZE: 10 * 1024 * 1024, // 10MB
      ACCEPTED_TYPES: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
      ACCEPTED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
};

// Decision Configuration with Colors and Descriptions
export const DECISION_CONFIG = {
      AUTO_APPROVE: {
            label: 'Auto Approved',
            color: '#222831',
            bg: '#76ABAE',
            borderColor: '#76ABAE',
            textColor: '#222831',
            description: 'Image appears authentic. No review needed.',
            severity: 'success',
      },
      FAST_TRACK: {
            label: 'Fast Track',
            color: '#EEEEEE',
            bg: '#31363F',
            borderColor: '#76ABAE',
            textColor: '#EEEEEE',
            description: 'Likely authentic. Minor review recommended.',
            severity: 'warning',
      },
      SUSPICIOUS: {
            label: 'Suspicious',
            color: '#EEEEEE',
            bg: '#31363F',
            borderColor: '#76ABAE',
            textColor: '#EEEEEE',
            description: 'Potential issues detected. Requires manual review.',
            severity: 'alert',
      },
      REJECT: {
            label: 'Rejected',
            color: '#EEEEEE',
            bg: '#222831',
            borderColor: '#76ABAE',
            textColor: '#EEEEEE',
            description: 'Image flagged as inauthentic. Do not approve.',
            severity: 'danger',
      },
};

// Layer Configuration
export const LAYER_CONFIG = {
      cnn: {
            name: 'CNN',
            fullName: 'Convolutional Neural Network',
            description: 'Detects low-level visual artifacts and pixel manipulation',
            icon: 'CONV',
      },
      vit: {
            name: 'ViT',
            fullName: 'Vision Transformer',
            description: 'Analyzes high-level semantic inconsistencies',
            icon: 'TRANS',
      },
      gan: {
            name: 'GAN',
            fullName: 'GAN Detector',
            description: 'Detects AI-generated and deepfake patterns',
            icon: 'GAN',
      },
      ocr: {
            name: 'OCR',
            fullName: 'Optical Character Recognition',
            description: 'Verifies text authenticity and date consistency',
            icon: 'OCR',
      },
};

// UI Constants
export const UI_CONFIG = {
      SKELETON_LOADING_TIME: 1000, // ms
      ANIMATION_DURATION: 0.3, // seconds
      TOAST_DURATION: 3000, // ms
};
