import React from 'react';
import { AlertOctagon, Loader2 } from 'lucide-react';
import { ApiError, ERROR_INFO } from '../api';

export function Loading({ text = 'Working…', progress }) {
  return (
    <div className="loading"><Loader2 size={18} className="spin" /><span>{text}</span>
      {progress !== undefined && <div className="progress"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>}</div>
  );
}

export function ErrorBox({ error }) {
  const e = error instanceof ApiError ? error : new ApiError(error?.message || 'Unexpected error');
  const [title, hint] = ERROR_INFO[e.code] || ERROR_INFO.BACKEND_ERROR;
  return <div className="errorCard"><div className="errorHead"><AlertOctagon size={16} /><b>{title}</b><code>{e.code}</code></div><p>{e.message}</p><p className="hint">{hint}</p></div>;
}
