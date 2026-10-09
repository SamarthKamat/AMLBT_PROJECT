// Same-origin by default (Flask serves the build; Vite proxies /api in dev). VITE_API_URL overrides it.
export const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, { code = 'BACKEND_ERROR', retryable = false, status = 0, plan = null, details = null } = {}) {
    super(message);
    this.code = code;
    this.retryable = retryable;
    this.status = status;
    this.plan = plan;
    this.details = details;
  }
}

export async function apiFetch(path, options = {}, timeoutMs = 180000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(API + path, { ...options, signal: controller.signal });
    let data = null;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) {
      throw new ApiError(data?.error || data?.message || `Backend request failed (${response.status}).`, {
        code: data?.code || 'BACKEND_ERROR', retryable: !!data?.retryable, status: response.status, plan: data?.plan || null, details: data?.details || null,
      });
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === 'AbortError') throw new ApiError('The request timed out. Model training on a large dataset can take a while; retry, or ask for a smaller scope.', { code: 'BACKEND_ERROR', retryable: true });
    throw new ApiError('Cannot reach the UDI backend. Start Flask (python app.py in /backend) and retry.', { code: 'BACKEND_OFFLINE', retryable: true });
  } finally {
    clearTimeout(timer);
  }
}

export const postJson = (path, body) => apiFetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// Friendly explanation per structured backend error code.
export const ERROR_INFO = {
  GEMINI_UNAVAILABLE: ['Gemini is not available', 'The language model that interprets questions could not be reached, so no analysis was run. Nothing was guessed.'],
  GEMINI_MODEL_UNAVAILABLE: ['Gemini model not available', 'The configured GEMINI_MODEL cannot be used with this API key. Update backend/.env and restart the backend.'],
  INVALID_PLAN: ['The question could not be turned into a valid analysis', 'Try rephrasing it, or name the column you care about.'],
  INVALID_COLUMN: ['That column is not in your dataset', 'UDI never substitutes a different column. Pick one of the available columns.'],
  INSUFFICIENT_DATA: ['Not enough data for this analysis', 'The dataset does not meet the minimum requirements for a reliable result.'],
  UNSUPPORTED_OPERATION: ['This kind of question is not supported', 'UDI answers questions about the uploaded dataset using its ten analysis operations.'],
  MODEL_TRAINING_ERROR: ['Model training failed', 'The model could not be trained on this selection of data.'],
  DATASET_PARSE_ERROR: ['The file could not be read', 'Check that it is a valid CSV/XLSX file with a header row.'],
  FILE_TOO_LARGE: ['File too large', 'Reduce the number of rows or columns and upload again.'],
  BACKEND_ERROR: ['Backend error', 'Something unexpected happened on the server.'],
  BACKEND_OFFLINE: ['Backend offline', 'The UI cannot reach the Flask backend.'],
  SESSION_NOT_FOUND: ['Session expired', 'Upload your dataset again to continue.'],
  NO_DATASET: ['No dataset yet', 'Upload a dataset before asking questions.'],
  BAD_REQUEST: ['Invalid request', 'Check the input and try again.'],
};
