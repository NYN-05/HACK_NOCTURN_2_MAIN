import { useMemo, useState } from 'react'
import './App.css'

const endpointPath = '/api/v1/verify'

function formatDecision(decision) {
  return (decision || 'UNKNOWN').replaceAll('_', ' ')
}

function getTodayDateString() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const apiUrl = useMemo(() => {
    const base = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/$/, '')
    return base ? `${base}${endpointPath}` : endpointPath
  }, [])

  async function onSubmit(event) {
    event.preventDefault()
    if (!imageFile) {
      setError('Please upload an image before verification.')
      return
    }

    setError('')
    setResult(null)
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      const today = getTodayDateString()
      formData.append('order_date', today)
      formData.append('delivery_date', today)
      // Do not send mfg_date_claimed so OCR can infer manufacturing date from the product image.

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        let message = `Verification failed with status ${response.status}.`
        try {
          const body = await response.json()
          if (body?.detail) {
            message = body.detail
          }
        } catch {
          // Keep fallback message if response body is not JSON.
        }
        throw new Error(message)
      }

      const payload = await response.json()
      setResult(payload)
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unexpected error during verification.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <p className="eyebrow">VeriSight</p>
        <h1>Authenticity Verification Console</h1>
        <p className="hero-copy">
          This UI sends only the product image. Order and delivery dates are auto-set to today, while
          manufacturing date is determined by the OCR model.
        </p>
      </section>

      <section className="grid-layout">
        <form className="card form-card" onSubmit={onSubmit}>
          <h2>Input</h2>
          <label className="field">
            <span>Image File</span>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={(event) => setImageFile(event.target.files?.[0] || null)}
              required
            />
          </label>

          <button type="submit" className="verify-btn" disabled={isLoading}>
            {isLoading ? 'Verifying...' : 'Run Verification'}
          </button>

          <p className="api-note">
            Endpoint: <code>{apiUrl}</code>
          </p>

          {error ? <p className="error-text">{error}</p> : null}
        </form>

        <section className="card result-card">
          <h2>Output</h2>
          {!result ? (
            <p className="placeholder">No result yet. Submit an image to calculate score and decision.</p>
          ) : (
            <>
              <div className="score-tile">
                <p>Final Score</p>
                <strong>{Number(result.authenticity_score || 0).toFixed(2)}</strong>
              </div>

              <div className="summary-row">
                <div>
                  <p className="key">Decision</p>
                  <p className="value">{formatDecision(result.decision)}</p>
                </div>
                <div>
                  <p className="key">Processing Time</p>
                  <p className="value">{result.processing_time_ms ?? 0} ms</p>
                </div>
              </div>

              <div>
                <p className="key">Layer Scores</p>
                <ul className="layer-list">
                  {Object.entries(result.layer_scores || {}).map(([layer, score]) => (
                    <li key={layer}>
                      <span>{layer.toUpperCase()}</span>
                      <strong>{Number(score).toFixed(2)}</strong>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  )
}

export default App
