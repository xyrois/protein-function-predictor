import type { PredictionResponse } from '../types/prediction';

const API_URL = import.meta.env.VITE_API_URL;

export async function predictSequence(sequence: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sequence }),
  });

  const data = await response.json();

  if (!response.ok) {
    const errorMsg =
      typeof data.detail === 'string'
        ? data.detail
        : data.detail?.[0]?.msg || 'Prediction request failed.';
    throw new Error(errorMsg);
  }

  return data;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}