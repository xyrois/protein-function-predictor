import React from 'react';
import { Play, Sparkles, Trash2 } from 'lucide-react';
import type { PresetSample } from '../types/prediction';

const SAMPLE_PRESETS: PresetSample[] = [
  {
    name: 'Lysozyme Fragment',
    expectedClass: 'Hydrolase',
    sequence: 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL'
  },
  {
    name: 'Alcohol Dehydrogenase',
    expectedClass: 'Oxidoreductase',
    sequence: 'MSSAAMTRAVIWEAERPLTIEEIEVDLPRGAGEVLVRIQATGVCHTDLHALDGEWPVPIKVPLVPGHEIVGTVVEVGEGVTKFKPGDRVGVGWLGNGCGTCEYCLSGNETLCPSGFSYTGYD'
  },
  {
    name: 'Glutamine Synthetase',
    expectedClass: 'Ligase',
    sequence: 'MTTASTEKILAEFAEYLQKYGIPVVDFRVEPWEWDDAPQAYVLEFIKDCGAEIEVVGFPFSDPYLEKIGKRLVEAGADAVVADLTVDREKA'
  }
];

interface SequenceInputProps {
  sequence: string;
  setSequence: (seq: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  error: string | null;
}

export const SequenceInput: React.FC<SequenceInputProps> = ({
  sequence,
  setSequence,
  onSubmit,
  isLoading,
  error,
}) => {
  const cleanLength = sequence.replace(/[\s\r\n]+/g, '').length;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <label className="text-sm font-medium text-slate-300">
          Protein Sequence (Single-letter FASTA / IUPAC)
        </label>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 font-mono">
            Length: {cleanLength} aa (Max: 512)
          </span>
          {sequence && (
            <button
              onClick={() => setSequence('')}
              className="text-xs text-slate-400 hover:text-rose-400 flex items-center gap-1 transition-colors"
            >
              <Trash2 className="w-3 h-3" /> Clear
            </button>
          )}
        </div>
      </div>

      <textarea
        rows={5}
        value={sequence}
        onChange={(e) => setSequence(e.target.value)}
        placeholder="Paste amino acid sequence here (e.g. MKTVRQERLKSIVRILERSKE...)"
        className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-3 font-mono text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-y uppercase"
      />

      {error && (
        <div className="mt-3 p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs font-mono">
          {error}
        </div>
      )}

      <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Presets:
          </span>
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => setSequence(p.sequence)}
              className="text-xs px-2.5 py-1 rounded-lg border border-slate-800 bg-slate-800/40 text-slate-300 hover:bg-slate-800 hover:border-slate-700 transition"
            >
              {p.name}
            </button>
          ))}
        </div>

        <button
          onClick={onSubmit}
          disabled={isLoading || cleanLength < 10}
          className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm text-white flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-600/20"
        >
          <Play className="w-4 h-4 fill-white" />
          {isLoading ? 'Running Inference...' : 'Predict Function'}
        </button>
      </div>
    </div>
  );
};