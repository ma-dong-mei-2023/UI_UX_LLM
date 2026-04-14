"use client";

interface Token {
  token: string;
  id: number;
}

const COLORS = [
  "bg-purple-500/30 border-purple-500/50 text-purple-200",
  "bg-blue-500/30 border-blue-500/50 text-blue-200",
  "bg-green-500/30 border-green-500/50 text-green-200",
  "bg-yellow-500/30 border-yellow-500/50 text-yellow-200",
  "bg-pink-500/30 border-pink-500/50 text-pink-200",
  "bg-orange-500/30 border-orange-500/50 text-orange-200",
  "bg-teal-500/30 border-teal-500/50 text-teal-200",
];

interface Props {
  tokens: Token[];
  label?: string;
}

export default function TokenHighlighter({ tokens, label }: Props) {
  if (!tokens.length) return null;
  return (
    <div>
      {label && <div className="text-xs text-gray-500 mb-1">{label}</div>}
      <div className="flex flex-wrap gap-1 p-3 bg-gray-900 rounded-lg border border-gray-700">
        {tokens.map((tok, i) => (
          <span
            key={i}
            className={`inline-flex flex-col items-center px-1.5 py-0.5 rounded border text-xs font-mono ${COLORS[i % COLORS.length]}`}
            title={`ID: ${tok.id}`}
          >
            <span className="whitespace-pre">{tok.token || " "}</span>
            <span className="text-[9px] opacity-60 mt-0.5">{tok.id}</span>
          </span>
        ))}
      </div>
      <div className="text-xs text-gray-500 mt-1">{tokens.length} tokens</div>
    </div>
  );
}
