export function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    const abs = Math.abs(value);
    if (abs !== 0 && abs < 0.001) return value.toExponential(2);
    return value.toLocaleString(undefined, { maximumFractionDigits: abs >= 1000 ? 0 : digits });
  }
  if (Array.isArray(value)) return value.map((v) => fmt(v, digits)).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export const label = (key) => String(key).replaceAll('_', ' ');
