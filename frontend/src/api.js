// Same-origin by default (Flask serves the build; Vite proxies /api in dev). VITE_API_URL overrides it.
export const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

const TOKEN_KEY = 'udi.token';
export function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; } }
export function setToken(t) { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* storage unavailable */ } }

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

function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

export async function apiFetch(path, options = {}, timeoutMs = 300000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(API + path, { ...options, headers: authHeaders(options.headers || {}), signal: controller.signal });
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
    throw new ApiError('Cannot reach the UDI backend. Start it (python serve.py in /backend) and retry.', { code: 'BACKEND_OFFLINE', retryable: true });
  } finally {
    clearTimeout(timer);
  }
}

export const getJson = (path) => apiFetch(path);
export const postJson = (path, body) => apiFetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
export const del = (path) => apiFetch(path, { method: 'DELETE' });

// Download a file from an authenticated endpoint (PDF / Excel / CSV).
export async function download(path, fallbackName) {
  const res = await fetch(API + path, { headers: authHeaders() });
  if (!res.ok) {
    let msg = `Download failed (${res.status}).`;
    try { msg = (await res.json()).error || msg; } catch { /* not JSON */ }
    throw new ApiError(msg, { status: res.status });
  }
  const blob = await res.blob();
  const cd = res.headers.get('Content-Disposition') || '';
  const m = /filename="?([^";]+)"?/.exec(cd);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = m ? m[1] : fallbackName;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

export function toCsv(rows) {
  if (!rows?.length) return '';
  const keys = Object.keys(rows[0]);
  const esc = (v) => { const s = v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [keys.join(','), ...rows.map((r) => keys.map((k) => esc(r[k])).join(','))].join('\n');
}

export function saveText(text, name, type = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// Friendly explanation per structured backend error code.
export const ERROR_INFO = {
  GEMINI_UNAVAILABLE: ['Gemini is not available', 'The language model could not be reached. In auto mode UDI falls back to its local planner; in Gemini-only mode nothing is guessed.'],
  GEMINI_MODEL_UNAVAILABLE: ['Gemini model not available', 'The configured GEMINI_MODEL cannot be used with this API key. Update backend/.env and restart the backend.'],
  INVALID_PLAN: ['The question could not be turned into a valid analysis', 'Try rephrasing it, or name the column you care about.'],
  INVALID_COLUMN: ['That column is not in your dataset', 'UDI never substitutes a different column. Pick one of the available columns.'],
  INSUFFICIENT_DATA: ['Not enough data for this analysis', 'The dataset does not meet the minimum requirements for a reliable result.'],
  UNSUPPORTED_OPERATION: ['This kind of question is not supported', 'UDI answers questions about the uploaded dataset using its eighteen analysis operations.'],
  MODEL_TRAINING_ERROR: ['Model training failed', 'The model could not be trained on this selection of data.'],
  DATASET_PARSE_ERROR: ['The file could not be read', 'Check that it is a valid CSV/XLSX file with a header row.'],
  FILE_TOO_LARGE: ['File too large', 'Reduce the number of rows or columns and upload again.'],
  BACKEND_ERROR: ['Backend error', 'Something unexpected happened on the server.'],
  BACKEND_OFFLINE: ['Backend offline', 'The UI cannot reach the Flask backend.'],
  SESSION_NOT_FOUND: ['Session expired', 'Upload your dataset again to continue.'],
  NO_DATASET: ['No dataset yet', 'Upload a dataset before asking questions.'],
  BAD_REQUEST: ['Invalid request', 'Check the input and try again.'],
  UNAUTHORIZED: ['Please sign in', 'Your session is not signed in or has expired.'],
  FORBIDDEN: ['Not allowed', 'This resource belongs to another user.'],
  RATE_LIMITED: ['Slow down a little', 'Too many requests in a short time. Wait a few seconds and retry.'],
  JOB_NOT_FOUND: ['Report not found', 'The report job has expired. Generate it again.'],
};
