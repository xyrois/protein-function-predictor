import React from 'react';
import { Dna, Activity } from 'lucide-react';

interface NavbarProps {
  isBackendConnected: boolean | null;
}

export const Navbar: React.FC<NavbarProps> = ({ isBackendConnected }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Dna className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold text-slate-100 leading-none">ESM-2 Enzyme Predictor</h1>
            <p className="text-xs text-slate-400 mt-1">EC Broad Functional Classification</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-full border border-slate-800 bg-slate-900">
          <Activity
            className={`w-3.5 h-3.5 ${
              isBackendConnected === true
                ? 'text-emerald-400 animate-pulse'
                : isBackendConnected === false
                ? 'text-rose-400'
                : 'text-amber-400'
            }`}
          />
          <span className="text-slate-300">
            {isBackendConnected === true
              ? 'Backend Online'
              : isBackendConnected === false
              ? 'Backend Disconnected'
              : 'Connecting...'}
          </span>
        </div>
      </div>
    </header>
  );
};