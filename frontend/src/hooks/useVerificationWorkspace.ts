import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getVerificationEndpointLabel, verifyImage } from '../lib/api';
import type { RunHistoryItem, VerificationResult } from '../types';

export function useVerificationWorkspace() {
  const requestIdRef = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [history, setHistory] = useState<RunHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const endpointLabel = useMemo(() => getVerificationEndpointLabel(), []);
  const apiModeLabel = useMemo(() => (import.meta.env.VITE_USE_MOCK_API === 'true' ? 'Mock fallback' : 'Live API'), []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);

    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const handleSelectFile = useCallback((nextFile: File | null) => {
    requestIdRef.current += 1;
    setError(null);
    setIsLoading(false);
    setFile(nextFile);
    setResult(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) {
      setError('Select an image before running verification.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    const requestId = ++requestIdRef.current;

    try {
      const nextResult = await verifyImage(file);
      if (requestId !== requestIdRef.current) {
        return;
      }
      setResult(nextResult);
      setHistory((currentHistory) => [
        {
          ...nextResult,
          fileName: file.name,
          fileSize: file.size,
        },
        ...currentHistory,
      ]);
    } catch (analysisError) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      const message = analysisError instanceof Error ? analysisError.message : 'Verification failed.';
      setError(message);
    } finally {
      if (requestId === requestIdRef.current) {
        setIsLoading(false);
      }
    }
  }, [file]);

  const analysisState = useMemo(() => {
    if (isLoading) {
      return 'Analyzing image';
    }

    if (file) {
      return 'Image loaded';
    }

    return 'Waiting for upload';
  }, [file, isLoading]);

  const clearSelection = useCallback(() => {
    requestIdRef.current += 1;
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    analysisState,
    apiModeLabel,
    endpointLabel,
    error,
    file,
    history,
    isLoading,
    handleAnalyze,
    handleSelectFile,
    clearSelection,
    previewUrl,
    result,
  };
}
