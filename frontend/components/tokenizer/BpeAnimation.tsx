"use client";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface TokenSample {
  id: number;
  s: string;
}

interface VocabEntry {
  id: number;
  token: string;
}

interface BpeStep {
  step: number;
  merged_pair: [number, number] | null;
  merged_pair_str: [string, string] | null;
  new_token_id: number | null;
  new_token: string | null;
  current_vocab_size: number;
  total_tokens: number;
  compression_ratio: number;
  tokens_sample: TokenSample[];
  full_vocab?: VocabEntry[];
}

interface Props {
  steps: BpeStep[];
  currentStep: number;
}

const COLORS = [
  "bg-purple-500/20 border-purple-500/40 text-purple-200",
  "bg-blue-500/20 border-blue-500/40 text-blue-200",
  "bg-green-500/20 border-green-500/40 text-green-200",
  "bg-yellow-500/20 border-yellow-500/40 text-yellow-200",
  "bg-pink-500/20 border-pink-500/40 text-pink-200",
  "bg-orange-500/20 border-orange-500/40 text-orange-200",
  "bg-teal-500/20 border-teal-500/40 text-teal-200",
];

export default function BpeAnimation({ steps, currentStep }: Props) {
  const stepData = steps[currentStep];
  const nextStepData = steps[currentStep + 1];
  
  // Find which pairs in the current tokens_sample will be merged in the next step
  const nextMergePair = nextStepData?.merged_pair;
  
  if (!stepData) return null;

  return (
    <div className="space-y-6">
      {/* Animation Canvas */}
      <div className="space-y-2">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">文本序列预览</h3>
        <Card className="bg-gray-950 border-gray-800 p-6 min-h-[200px] flex flex-wrap gap-1 items-start content-start overflow-auto max-h-[400px]">
          {stepData.tokens_sample.map((tok, i) => {
            // Check if this token and the next one form the pair to be merged
            const isPartOfNextMerge = 
              nextMergePair && 
              tok.id === nextMergePair[0] && 
              stepData.tokens_sample[i+1]?.id === nextMergePair[1];
            
            const isSecondPartOfNextMerge = 
              nextMergePair && 
              tok.id === nextMergePair[1] && 
              stepData.tokens_sample[i-1]?.id === nextMergePair[0];

            let extraClass = "";
            if (isPartOfNextMerge) extraClass = "ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-950 z-10 scale-110 translate-x-0.5";
            if (isSecondPartOfNextMerge) extraClass = "ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-950 z-10 scale-110 -translate-x-0.5";

            return (
              <div
                key={`${currentStep}-${i}`}
                className={`
                  relative flex flex-col items-center px-2 py-1 rounded border text-sm font-mono transition-all duration-300
                  ${COLORS[tok.id % COLORS.length]}
                  ${extraClass}
                `}
                title={`ID: ${tok.id}`}
              >
                <span className="whitespace-pre">{tok.s || " "}</span>
                <span className="text-[10px] opacity-40 mt-0.5">{tok.id}</span>
              </div>
            );
          })}
          {stepData.total_tokens > stepData.tokens_sample.length && (
            <div className="text-gray-600 text-xs self-end pb-2 ml-2">... 以及另外 {stepData.total_tokens - stepData.tokens_sample.length} 个 Token</div>
          )}
        </Card>
      </div>

      {/* Explanation Area */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-gray-900/50 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-2">💡 步骤解析</h3>
          <div className="text-sm text-gray-400 leading-relaxed">
            {nextStepData ? (
              <>
                当前状态下，系统发现相邻的 
                <Badge variant="outline" className="mx-1 text-yellow-400 border-yellow-400/50 font-mono">"{nextStepData.merged_pair_str?.[0]}"</Badge> 
                和 
                <Badge variant="outline" className="mx-1 text-yellow-400 border-yellow-400/50 font-mono">"{nextStepData.merged_pair_str?.[1]}"</Badge>
                出现的频率最高。点击“下一步”将把它们合并为新的 Token 
                <Badge className="mx-1 bg-purple-600 font-mono">"{nextStepData.new_token}"</Badge>。
              </>
            ) : (
              "训练已完成！所有最高频的相邻对都已根据目标词汇表大小完成了合并。"
            )}
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-500/30 rounded-lg p-4 flex flex-col justify-center items-center text-center">
          <div className="text-xs text-purple-300/70 mb-1 uppercase tracking-wider font-bold">压缩效果</div>
          <div className="text-3xl font-bold text-purple-400 font-mono">{stepData.compression_ratio}x</div>
          <div className="text-[10px] text-purple-300/50 mt-1">原始长度 / 当前 Token 数</div>
        </div>
      </div>

      {/* Full Vocabulary Display (only at the end) */}
      {stepData.full_vocab && !nextStepData && (
        <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-300">📚 完整词汇表 ({stepData.full_vocab.length})</h3>
            <span className="text-[10px] text-gray-500">ID 0-255 为原始字节，256+ 为合并后的新 Token</span>
          </div>
          <Card className="bg-gray-900 border-gray-800 overflow-hidden">
            <ScrollArea className="h-[300px] w-full p-4">
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                {stepData.full_vocab.map((entry) => (
                  <div 
                    key={entry.id} 
                    className={`
                      flex items-center justify-between px-3 py-1.5 rounded text-xs font-mono border
                      ${entry.id < 256 
                        ? "bg-gray-800/30 border-gray-700/50 text-gray-400" 
                        : "bg-purple-500/10 border-purple-500/20 text-purple-200"}
                    `}
                  >
                    <span className="truncate mr-2" title={entry.token}>{entry.token || " "}</span>
                    <span className="opacity-40 text-[9px]">{entry.id}</span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </Card>
        </div>
      )}
    </div>
  );
}

