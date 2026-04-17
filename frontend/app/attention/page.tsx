"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const DEFAULT_TOKENS = ["The", " quick", " brown", " fox", " jumps"];
const ATTENTION_TYPES = ["simple", "causal", "multihead"] as const;
type AttentionType = typeof ATTENTION_TYPES[number];
type StepType = "raw_scores" | "scaled_scores" | "masked_scores" | "weights";

interface InspectorData {
  q: number[];
  k: number[];
  rowIdx: number;
  colIdx: number;
  rowToken: string;
  colToken: string;
  rawScore: number;
}

function AttentionHeatmap({ 
  tokens, 
  matrix, 
  step,
  onCellClick,
  selectedCell
}: {
  tokens: string[];
  matrix: number[][];
  step: StepType;
  onCellClick?: (r: number, c: number) => void;
  selectedCell?: {r: number, c: number} | null;
}) {
  if (!matrix || !matrix.length) return null;

  // Filter out -inf for min/max calculation
  const validVals = matrix.flat().filter(v => v > -1000);
  const max = validVals.length ? Math.max(...validVals) : 1;
  const min = validVals.length ? Math.min(...validVals) : 0;

  return (
    <div className="overflow-auto pb-4">
      <table className="text-xs font-mono border-collapse relative">
        <thead>
          <tr>
            <th className="w-20 text-right pr-2 text-gray-500">
              <div className="text-[10px] text-gray-600 mb-[-4px]">Q (当前) ↓</div>
              <div>K (被看) →</div>
            </th>
            {tokens.map((t, i) => (
              <th key={i} className="px-1 text-gray-400 font-normal max-w-[60px] truncate pb-2" title={t}>
                {t.trim().slice(0, 8)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className="hover:bg-gray-800/50 group">
              <td className="text-right pr-2 text-gray-400 truncate max-w-[80px] group-hover:text-gray-200 transition-colors">
                {tokens[i]?.trim().slice(0, 8)}
              </td>
              {row.map((val, j) => {
                const isMasked = val < -1000;
                let intensity = 0;
                let displayVal = "";
                
                if (isMasked) {
                  displayVal = "×";
                } else {
                  if (step === "weights") {
                    intensity = val;
                    displayVal = val.toFixed(2);
                  } else {
                    // Normalize for other steps
                    intensity = max === min ? 0.5 : (val - min) / (max - min);
                    displayVal = val.toFixed(1);
                  }
                }

                // Color mapping: Red for negative, Blue for positive, dark for masked
                const isNegative = val < 0 && !isMasked;
                let bg = "rgb(17,24,39)"; // default dark
                
                if (isMasked) {
                  bg = "rgb(40,20,20)";
                } else if (step === "weights") {
                  const r = Math.round(intensity * 59 + (1 - intensity) * 17);
                  const g = Math.round(intensity * 130 + (1 - intensity) * 24);
                  const b = Math.round(intensity * 246 + (1 - intensity) * 39);
                  bg = `rgb(${r},${g},${b})`;
                } else {
                  if (isNegative) {
                    const negInt = Math.min(1, Math.abs(val) / Math.max(1, Math.abs(min)));
                    bg = `rgba(239,68,68,${negInt * 0.8})`; // Redish
                  } else {
                    const posInt = Math.min(1, val / Math.max(1, max));
                    bg = `rgba(59,130,246,${posInt * 0.8})`; // Blueish
                  }
                }

                const isSelected = selectedCell?.r === i && selectedCell?.c === j;

                return (
                  <td
                    key={j}
                    onClick={() => !isMasked && onCellClick?.(i, j)}
                    className={`
                      w-12 h-10 text-center rounded border transition-all
                      ${isMasked ? 'cursor-not-allowed' : 'cursor-pointer hover:ring-2 hover:ring-white/50'}
                      ${isSelected ? 'ring-2 ring-yellow-400 z-10 scale-110 shadow-lg' : 'border-gray-800/50'}
                    `}
                    style={{ backgroundColor: bg, color: (step === "weights" ? intensity > 0.5 : Math.abs(val) > (max-min)*0.4) ? "#fff" : "#aaa" }}
                    title={isMasked ? "因果掩码：不能看未来的词" : val.toFixed(4)}
                  >
                    {displayVal}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DotProductInspector({ data }: { data: InspectorData | null }) {
  if (!data) return (
    <div className="text-sm text-gray-500 text-center py-8">
      点击上方热力图（非掩码区域）的一个单元格，查看点积计算过程
    </div>
  );

  const { q, k, rowToken, colToken, rawScore } = data;
  const d = q.length;
  
  // Render a mini vector
  const renderVector = (vec: number[], label: string, name: string) => {
    const maxVal = Math.max(...vec.map(Math.abs));
    return (
      <div className="flex flex-col gap-1 mb-3">
        <div className="text-xs text-gray-400 flex justify-between">
          <span>{name}: <span className="text-gray-200 font-medium">"{label.trim()}"</span></span>
          <span className="text-[10px]">维度: {d}</span>
        </div>
        <div className="flex flex-wrap gap-[1px]">
          {vec.slice(0, 32).map((v, i) => {
            const int = Math.abs(v) / (maxVal || 1);
            const isNeg = v < 0;
            return (
              <div 
                key={i} 
                className="w-3 h-4 rounded-sm"
                style={{ backgroundColor: isNeg ? `rgba(239,68,68,${int})` : `rgba(59,130,246,${int})` }}
                title={`dim ${i}: ${v.toFixed(3)}`}
              />
            )
          })}
          {d > 32 && <span className="text-[10px] text-gray-500 ml-1 leading-4">...</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="animate-in fade-in slide-in-from-top-4 duration-300">
      <div className="flex items-center gap-2 mb-4 text-sm">
        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">探视镜</Badge>
        <span className="text-gray-300">正在解析：</span>
        <span className="font-mono bg-gray-800 px-1.5 py-0.5 rounded text-blue-400">{rowToken.trim()}</span>
        <span className="text-gray-500">对</span>
        <span className="font-mono bg-gray-800 px-1.5 py-0.5 rounded text-blue-400">{colToken.trim()}</span>
        <span className="text-gray-300">的关注度</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-950/50 p-4 rounded-lg border border-gray-800">
        <div>
          {renderVector(q, rowToken, "Query 向量 (寻找)")}
          <div className="text-center text-gray-600 my-1 text-lg leading-none">×</div>
          {renderVector(k, colToken, "Key 向量 (提供)")}
        </div>
        
        <div className="flex flex-col justify-center border-l border-gray-800 pl-6">
          <div className="text-xs text-gray-400 mb-2">点积求和 (Dot Product) = Σ (Q_i × K_i)</div>
          <div className="text-3xl font-mono text-white mb-2">{rawScore.toFixed(3)}</div>
          <p className="text-xs text-gray-500 leading-relaxed">
            这个原始得分会被除以 <span className="text-gray-300 font-mono">√{d}</span> 进行缩放，
            然后再通过 <span className="text-gray-300 font-mono">Softmax</span> 转化为 0~1 之间的最终注意力权重。
          </p>
        </div>
      </div>
    </div>
  );
}

export default function AttentionPage() {
  const [tokenInput, setTokenInput] = useState(DEFAULT_TOKENS.join(", "));
  const [dModel, setDModel] = useState(64);
  const [nHeads, setNHeads] = useState(4);
  const [mode, setMode] = useState("toy_previous");
  const [activeHead, setActiveHead] = useState(0);
  const [activeStep, setActiveStep] = useState<StepType>("weights");
  const [gridView, setGridView] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{r: number, c: number} | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Record<AttentionType, any>>({
    simple: null, causal: null, multihead: null,
  });
  const [activeType, setActiveType] = useState<AttentionType>("simple");

  const tokens = tokenInput.split(",").map((t) => t.trim()).filter(Boolean);

  async function runAttention(type: AttentionType) {
    if (tokens.length < 2) { setError("至少需要 2 个 token"); return; }
    setLoading(true); setError("");
    setSelectedCell(null);
    try {
      let result;
      if (type === "simple") result = await api.simpleAttention(tokens, dModel, nHeads, mode);
      else if (type === "causal") result = await api.causalAttention(tokens, dModel, nHeads, mode);
      else result = await api.multiheadAttention(tokens, dModel, nHeads, mode);
      setResults((prev) => ({ ...prev, [type]: result }));
      setActiveType(type);
      
      // Auto-select step based on availability
      if (activeStep === "masked_scores" && !result.masked_scores) {
        setActiveStep("weights");
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runAll() {
    if (tokens.length < 2) { setError("至少需要 2 个 token"); return; }
    setLoading(true); setError("");
    setSelectedCell(null);
    try {
      const [s, c, m] = await Promise.all([
        api.simpleAttention(tokens, dModel, nHeads, mode),
        api.causalAttention(tokens, dModel, nHeads, mode),
        api.multiheadAttention(tokens, dModel, nHeads, mode),
      ]);
      setResults({ simple: s, causal: c, multihead: m });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const currentResult = results[activeType];
  
  // Prepare inspector data
  let inspectorData: InspectorData | null = null;
  if (currentResult && selectedCell && currentResult.q && currentResult.k && currentResult.raw_scores) {
    const { r, c } = selectedCell;
    const isMulti = activeType === "multihead";
    
    // Extract vectors for the specific head if multihead
    let qVec, kVec, raw;
    if (isMulti) {
      // q shape: [heads, seq, d_head]
      qVec = currentResult.q[activeHead][r];
      kVec = currentResult.k[activeHead][c];
      raw = currentResult.raw_scores[activeHead][r][c];
    } else {
      qVec = currentResult.q[r];
      kVec = currentResult.k[c];
      raw = currentResult.raw_scores[r][c];
    }

    if (qVec && kVec) {
      inspectorData = {
        q: qVec, k: kVec, rowIdx: r, colIdx: c,
        rowToken: tokens[r], colToken: tokens[c], rawScore: raw
      };
    }
  }

  // Helper to extract the right matrix for the current step & head
  const getMatrix = (headIdx: number = 0) => {
    if (!currentResult || !currentResult[activeStep]) return [];
    
    // For masked scores on simple attention, fallback to scaled
    if (activeStep === "masked_scores" && !currentResult.masked_scores) {
      return activeType === "multihead" 
        ? currentResult.scaled_scores[headIdx] 
        : currentResult.scaled_scores;
    }

    return activeType === "multihead" 
      ? currentResult[activeStep][headIdx] 
      : currentResult[activeStep];
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">👁️ 注意力可视化</h1>
        <p className="text-gray-400 text-sm">
          第 3 章 · 拆解 Transformer 的核心计算过程，揭开 Q×K 点积与多头机制的黑盒。
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
        {/* Config Sidebar */}
        <div className="space-y-4">
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">数据源配置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <div>
                <label className="text-xs text-gray-400 block mb-1">教学模式</label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="w-full bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none"
                >
                  <option value="toy_previous">💡 预设：前驱词关注 (推荐)</option>
                  <option value="random">🎲 随机初始化权重 (无意义)</option>
                </select>
                <p className="text-[10px] text-gray-500 mt-1 leading-tight">
                  {mode === 'toy_previous' 
                    ? "模拟真实训练后的特征：Head 0 强迫关注前一个词，Head 1 强迫关注句首。" 
                    : "使用随机初始化的 Embedding 和 Linear 层，产生的注意力是杂乱的。"}
                </p>
              </div>

              <div>
                <label className="text-xs text-gray-400 block mb-1 flex justify-between">
                  <span>输入句子（逗号分隔 Token）</span>
                  <span className="text-gray-600">{tokens.length} tokens</span>
                </label>
                <input
                  value={tokenInput}
                  onChange={(e) => setTokenInput(e.target.value)}
                  className="w-full bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none"
                  placeholder="The, quick, brown, fox"
                />
              </div>
              
              <Separator className="bg-gray-800" />

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-400">模型维度 (d_model)</span>
                  <span className="text-blue-400 font-mono">{dModel}</span>
                </div>
                <Slider value={[dModel]} onValueChange={(v) => { const n = Array.isArray(v) ? v[0] : v; setDModel(n); setActiveHead(0); }}
                  min={16} max={128} step={16} className="[&_[role=slider]]:bg-blue-500" />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-400">注意力头数 (n_heads)</span>
                  <span className="text-blue-400 font-mono">{nHeads}</span>
                </div>
                <Slider value={[nHeads]} onValueChange={(v) => { const n = Array.isArray(v) ? v[0] : v; setNHeads(n); setActiveHead(0); }}
                  min={1} max={8} step={1} className="[&_[role=slider]]:bg-blue-500" />
                <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                  <span>每头维度 (d_head): {dModel / nHeads}</span>
                </div>
              </div>

              {dModel % nHeads !== 0 && (
                <div className="text-xs text-red-400 bg-red-500/10 rounded p-2">
                  ⚠️ d_model 必须能被 n_heads 整除
                </div>
              )}

              {error && <p className="text-red-400 text-xs">{error}</p>}

              <button
                onClick={() => runAttention(activeType)}
                disabled={loading || dModel % nHeads !== 0}
                className="w-full py-2.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium shadow-lg shadow-blue-900/20 transition-all active:scale-95"
              >
                {loading ? "计算中..." : "重新计算"}
              </button>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Area */}
        <div className="space-y-4">
          
          {/* Type Selector */}
          <div className="flex gap-2 p-1 bg-gray-900 rounded-lg border border-gray-800 w-fit">
            {(["simple", "causal", "multihead"] as AttentionType[]).map((type) => {
              const labels = { simple: "1. 自注意力", causal: "2. 因果注意力", multihead: "3. 多头注意力" };
              return (
                <button
                  key={type}
                  onClick={() => { setActiveType(type); if (!results[type]) runAttention(type); }}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                    activeType === type
                      ? "bg-gray-800 text-white shadow-sm"
                      : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
                  }`}
                >
                  {labels[type]}
                  {results[type] && activeType !== type && <span className="ml-1.5 text-[10px] text-green-500">●</span>}
                </button>
              );
            })}
          </div>

          {currentResult ? (
            <div className="space-y-4 animate-in fade-in duration-300">
              
              {/* Step Tabs & Controls */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-gray-900 p-2 pl-4 rounded-lg border border-gray-800">
                <Tabs value={activeStep} onValueChange={(v) => setActiveStep(v as StepType)} className="w-full sm:w-auto">
                  <TabsList className="bg-gray-950/50 border border-gray-800/50">
                    <TabsTrigger value="raw_scores" className="text-xs">1. 原始点积 Q·K^T</TabsTrigger>
                    <TabsTrigger value="scaled_scores" className="text-xs">2. 缩放 /√d</TabsTrigger>
                    <TabsTrigger value="masked_scores" disabled={!currentResult.is_masked} className="text-xs">3. 掩码</TabsTrigger>
                    <TabsTrigger value="weights" className="text-xs">4. Softmax (最终)</TabsTrigger>
                  </TabsList>
                </Tabs>
                
                {activeType === "multihead" && currentResult.num_heads > 1 && (
                  <div className="flex items-center gap-3 pr-2">
                    <label className="text-xs text-gray-400 flex items-center gap-2 cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={gridView} 
                        onChange={(e) => setGridView(e.target.checked)}
                        className="rounded border-gray-700 bg-gray-800 text-blue-500 focus:ring-blue-500/50"
                      />
                      并排对比所有头
                    </label>
                    
                    {!gridView && (
                      <div className="flex gap-1">
                        {Array.from({ length: currentResult.num_heads }).map((_, h) => (
                          <button
                            key={h}
                            onClick={() => setActiveHead(h)}
                            className={`w-7 h-7 rounded text-xs font-mono transition-colors ${h === activeHead ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                          >
                            H{h}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Heatmaps Display */}
              {activeType === "multihead" && gridView ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Array.from({ length: currentResult.num_heads }).map((_, h) => (
                    <Card key={h} className="bg-gray-900 border-gray-800 overflow-hidden">
                      <div className="bg-gray-950 px-4 py-2 border-b border-gray-800 flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-300">Head {h}</span>
                        <span className="text-[10px] text-gray-500 font-mono">dim: {dModel/nHeads}</span>
                      </div>
                      <CardContent className="p-4 overflow-x-auto">
                        <AttentionHeatmap
                          tokens={currentResult.tokens}
                          matrix={getMatrix(h)}
                          step={activeStep}
                          onCellClick={(r, c) => { setActiveHead(h); setSelectedCell({r, c}); }}
                          selectedCell={h === activeHead ? selectedCell : null}
                        />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Card className="bg-gray-900 border-gray-800">
                  <CardContent className="p-6 overflow-x-auto">
                    <AttentionHeatmap
                      tokens={currentResult.tokens}
                      matrix={getMatrix(activeHead)}
                      step={activeStep}
                      onCellClick={(r, c) => setSelectedCell({r, c})}
                      selectedCell={selectedCell}
                    />
                  </CardContent>
                </Card>
              )}

              {/* Legend */}
              <div className="flex items-center gap-4 text-xs text-gray-500 px-2">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: activeStep === "weights" ? "rgb(17,24,39)" : "rgba(239,68,68,0.5)" }} />
                  <span>{activeStep === "weights" ? "低注意力 (0)" : "负相关 / 远离"}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: activeStep === "weights" ? "rgb(59,130,246)" : "rgba(59,130,246,0.8)" }} />
                  <span>{activeStep === "weights" ? "高注意力 (~1)" : "正相关 / 匹配"}</span>
                </div>
                {currentResult.is_masked && (
                  <div className="flex items-center gap-1.5 ml-4">
                    <div className="w-3 h-3 rounded bg-[#281414] border border-red-900/30 flex items-center justify-center text-[8px] text-red-500/50">×</div>
                    <span>被因果掩码阻挡 (-∞)</span>
                  </div>
                )}
              </div>

              {/* Inspector Panel */}
              <Card className="bg-gray-900 border-gray-800 mt-6 shadow-xl border-t-4 border-t-blue-600/30">
                <CardContent className="p-6">
                  <DotProductInspector data={inspectorData} />
                </CardContent>
              </Card>

            </div>
          ) : (
            <Card className="bg-gray-900 border-gray-800 border-dashed h-64 flex flex-col items-center justify-center text-gray-500 gap-3">
              <div className="text-4xl opacity-50">👆</div>
              <p>请点击上方按钮，运行注意力计算以查看结果</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}