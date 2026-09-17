import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { SequenceInput } from './components/SequenceInput';
import { PredictionCard } from './components/PredictionCard';
import { predictSequence, checkBackendHealth } from './services/api';
import type { PredictionResponse } from './types/prediction';

export default function App() {
  const [sequence, setSequence] = useState('');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);

  useEffect(() => {
    checkBackendHealth().then(setIsBackendConnected);
  }, []);

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await predictSequence(sequence);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <Navbar isBackendConnected={isBackendConnected} />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-8 space-y-6">
        <SequenceInput
          sequence={sequence}
          setSequence={setSequence}
          onSubmit={handlePredict}
          isLoading={isLoading}
          error={error}
        />

        {result && <PredictionCard result={result} />}
      </main>

      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-600">
        Protein Function Predictor • Powered by ESM-2 (facebook/esm2_t6_8M_UR50D) & FastAPI
      </footer>
    </div>
  );
}