import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react';
import { Button } from './Button';

interface UploadDropzoneProps {
  file: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File | null) => void;
  onAnalyze: () => void;
  onClear?: () => void;
  isLoading: boolean;
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp'];

function isAcceptedFile(file: File): boolean {
  return ACCEPTED_TYPES.includes(file.type);
}

function formatBytes(size: number): string {
  if (size === 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);
  return `${(size / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function UploadDropzone({ file, previewUrl, onFileSelect, onAnalyze, onClear, isLoading }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const openPicker = () => {
    inputRef.current?.click();
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    onFileSelect(nextFile && isAcceptedFile(nextFile) ? nextFile : null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const nextFile = event.dataTransfer.files?.[0] ?? null;
    onFileSelect(nextFile && isAcceptedFile(nextFile) ? nextFile : null);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      setIsDragging(false);
    }
  };

  return (
    <div
      className={`upload-dropzone ${isDragging ? 'is-dragging' : ''}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      aria-label="Upload image"
    >
      <div className="upload-dropzone__body">
        <p className="eyebrow">Upload</p>
        <h3>Drop an image here or choose a file</h3>
        <p>
          PNG, JPEG, WebP, GIF, and BMP files are supported. The preview appears immediately so the operator can
          confirm the file before sending it to the verification endpoint.
        </p>

        <ul className="upload-hints">
          <li>Preview before analysis</li>
          <li>Single image only</li>
          <li>Fast verification flow</li>
        </ul>
      </div>

      <label className="sr-only" htmlFor="verification-upload-input">
        Select image file
      </label>
      <input
        ref={inputRef}
        id="verification-upload-input"
        className="sr-only"
        type="file"
        accept="image/*"
        aria-label="Select image file"
        onChange={handleInputChange}
      />

      <div className="upload-dropzone__actions">
        <Button type="button" variant="secondary" onClick={openPicker}>
          Choose file
        </Button>
        <Button type="button" onClick={onAnalyze} disabled={!file || isLoading}>
          {isLoading ? 'Analyzing…' : 'Run verification'}
        </Button>
        {file && onClear ? (
          <Button type="button" variant="ghost" onClick={onClear}>
            Clear selection
          </Button>
        ) : null}
      </div>

      <div className="upload-preview">
        <div className="upload-preview__header">
          <div>
            <p className="eyebrow">Preview</p>
            <h3>{file ? file.name : 'No image selected'}</h3>
          </div>
          <span className={`badge ${file ? 'badge--success' : 'badge--neutral'}`}>{file ? 'Ready' : 'Waiting'}</span>
        </div>

        {previewUrl ? (
          <div className="upload-preview__frame">
            <img src={previewUrl} alt="Selected upload preview" className="upload-preview__image" />
          </div>
        ) : (
          <div className="upload-preview__empty">
            <p>Select an image to inspect it locally before analysis.</p>
          </div>
        )}

        <dl className="upload-meta">
          <div>
            <dt>Type</dt>
            <dd>{file?.type ?? 'Image file'}</dd>
          </div>
          <div>
            <dt>Size</dt>
            <dd>{file ? formatBytes(file.size) : '—'}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}