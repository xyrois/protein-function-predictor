import React from 'react';
import type { PredictionResponse } from '../types/prediction';
import { Award, AlertTriangle, Layers } from 'lucide-react';

const EC_DESCRIPTIONS: Record<string, string> = {
  Oxidoreductase: 'EC 1: Catalyzes oxidation-reduction reactions (electron transfer).',
  Transferase: 'EC 2: Catalyzes the transfer of functional groups between donor and acceptor.',
  Hydrolase: 'EC 3: Catalyzes the hydrolytic cleavage of chemical bonds (C-O, C-N, C-C).',
  Lyase: 'EC 4: Catalyzes cleavage of bonds without hydrolysis or oxidation, often leaving double bonds.',
  Isomerase: 'EC 5: Catalyzes geometric or structural changes within a single molecule.',
  Ligase: 'EC 6: Catalyzes joining of two molecules coupled with ATP/GTP hydrolysis.',
};

interface PredictionCardProps {
  result: PredictionResponse;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ result }) => {
  const topConfidence = (result.confidence * 100).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Primary Result Banner */}
      <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-br from-indigo-950/40 via-slate-900 to-slate-900 p-6 backdrop-blur-sm shadow-xl">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <span className="text-xs uppercase tracking-wider text-indigo-400 font-mono font-medium flex items-center gap-1.5">
              <Award className="w-4 h-4" /> Primary Predicted Class
            </span>
            <h2 className="text-3xl font-bold text-white mt-1">
              EC: {result.prediction}
            </h2>
            <p className="text-sm text-slate-400 mt-2 max-w-xl">
              {EC_DESCRIPTIONS[result.prediction] || 'Enzyme Commission Functional Class'}
            </p>
          </div>

          <div className="text-right">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-mono">Confidence</span>
            <div className="text-4xl font-extrabold text-indigo-400 font-mono mt-1">
              {topConfidence}%
            </div>
          </div>
        </div>
      </div>

      {/* Top 3 Distribution */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" /> Top-3 Class Probability Distribution
        </h3>

        <div className="space-y-4">
          {result.top_predictions.map((p, idx) => {
            const pct = (p.probability * 100).toFixed(2);
            return (
              <div key={p.ec_class} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-300">
                    <span className="text-slate-500 mr-2">#{idx + 1}</span>
                    {p.ec_class}
                  </span>
                  <span className="text-slate-400">{pct}%</span>
                </div>
                <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      idx === 0
                        ? 'bg-indigo-500'
                        : idx === 1
                        ? 'bg-slate-400'
                        : 'bg-slate-600'
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Disclaimer */}
        <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-start gap-2 text-xs text-amber-300/80">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
          <span>
            {result.disclaimer} Trained on ESM-2 8M with SwissProt-EC broad classes (Test F1: ~0.52).
          </span>
        </div>
      </div>
    </div>
  );
};