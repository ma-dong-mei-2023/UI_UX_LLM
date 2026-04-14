"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { sliderVal } from "@/lib/slider";
import type { GPTConfig, ParamCount, MemoryEstimate } from "@/lib/types";

const PRESETS: Record<string, GPTConfig> = {
  tiny: { vocab_size: 50257, context_length: 64, emb_dim: 48, n_heads: 2, n_layers: 2, drop_rate: 0.1, qkv_bias: false },
  small: { vocab_size: 50257, context_length: 128, emb_dim: 128, n_heads: 4, n_layers: 4, drop_rate: 0.1, qkv_bias: false },
  "gpt2-124m": { vocab_size: 50257, context_length: 256, emb_dim: 768, n_heads: 12, n_layers: 12, drop_rate: 0.1, qkv_bias: true },
  "gpt2-355m": { vocab_size: 50257, context_length: 1024, emb_dim: 1024, n_heads: 16, n_layers: 24, drop_rate: 0.1, qkv_bias: true },
};

const PRESET_LABELS: Record<string, string> = {
  tiny: "Tiny (~300K)", small: "Small (~3M)", "gpt2-124m": "GPT-2 124M", "gpt2-355m": "GPT-2 355M",
};

function formatNum(n: number) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toString();
}

function LayerRow({ name, type, shape, params }: { name: string; type: string; shape?: string; params?: number }) {
  const colors: Record<string, string> = {
    embedding: "bg-purple-500/20 border-purple-500/40 text-purple-300",
    attention: "bg-blue-500/20 border-blue-500/40 text-blue-300",
    norm: "bg-green-500/20 border-green-500/40 text-green-300",
    ff: "bg-orange-500/20 border-orange-500/40 text-orange-300",
    linear: "bg-gray-700/50 border-gray-600 text-gray-300",
    block: "bg-gray-800/80 border-gray-600 text-white",
    activation: "bg-yellow-500/10 border-yellow-500/30 text-yellow-300",
    dropout: "bg-gray-700/30 border-gray-600 text-gray-400",
    residual: "bg-red-500/10 border-red-500/20 text-red-400",
  };
  const cls = colors[type] || "bg-gray-800 border-gray-700 text-gray-300";
  return (
    <div className={`flex items-center justify-between px-3 py-1.5 rounded border text-xs font-mono ${cls}`}>
      <span>{name}</span>
      <div className="flex items-center gap-3 text-right">
        {shape && <span className="text-gray-500">{shape}</span>}
        {params != null && <span>{formatNum(params)}</span>}
      </div>
    </div>
  );
}

export default function ArchitecturePage() {
  const [cfg, setCfg] = useState<GPTConfig>(PRESETS["tiny"]);
  const [activePreset, setActivePreset] = useState("tiny");
  const [params, setParams] = useState<ParamCount | null>(null);
  const [memory, setMemory] = useState<MemoryEstimate | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedBlock, setExpandedBlock] = useState(false);

  useEffect(() => {
    fetchAnalysis();
  }, []);

  async function fetchAnalysis() {
    setLoading(true);
    try {
      const result: any = await api.buildModel(cfg);
      setParams(result.params);
      setMemory(result.memory);
      setErrors(result.errors || []);
    } catch (e: any) {
      setErrors([e.message]);
    } finally {
      setLoading(false);
    }
  }

  function applyPreset(name: string) {
    setCfg(PRESETS[name]);
    setActivePreset(name);
  }

  function updateCfg(key: keyof GPTConfig, value: number | boolean) {
    setCfg((prev) => ({ ...prev, [key]: value }));
    setActivePreset("");
  }

  const headDim = cfg.emb_dim / cfg.n_heads;
  const validHead = cfg.emb_dim % cfg.n_heads === 0;

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">🏗️ 架构构建器</h1>
        <p className="text-gray-400 text-sm">第 4 章 · 配置 GPT 模型参数，实时查看参数量、显存占用和网络结构。</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="xl:col-span-1 space-y-4">
          {/* Presets */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">预设配置</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
              {Object.entries(PRESET_LABELS).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => applyPreset(key)}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition-colors ${
                    activePreset === key
                      ? "bg-green-600 text-white"
                      : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                  }`}
                >
                  {label}
                </button>
              ))}
            </CardContent>
          </Card>

          {/* Sliders */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">模型参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { key: "emb_dim" as const, label: "嵌入维度", min: 16, max: 1024, step: 16, unit: "" },
                { key: "n_heads" as const, label: "注意力头数", min: 1, max: 32, step: 1, unit: "" },
                { key: "n_layers" as const, label: "Transformer 层数", min: 1, max: 48, step: 1, unit: "" },
                { key: "context_length" as const, label: "上下文长度", min: 32, max: 2048, step: 32, unit: "" },
              ].map(({ key, label, min, max, step }) => (
                <div key={key}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-gray-400">{label}</span>
                    <span className="text-green-400 font-mono">{cfg[key]}</span>
                  </div>
                  <Slider
                    value={[cfg[key] as number]}
                    onValueChange={(v) => updateCfg(key, sliderVal(v))}
                    min={min} max={max} step={step}
                    className="[&_[role=slider]]:bg-green-500"
                  />
                </div>
              ))}

              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">QKV 偏置</span>
                <button
                  onClick={() => updateCfg("qkv_bias", !cfg.qkv_bias)}
                  className={`w-10 h-5 rounded-full transition-colors ${cfg.qkv_bias ? "bg-green-600" : "bg-gray-700"}`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white mx-0.5 transition-transform ${cfg.qkv_bias ? "translate-x-5" : ""}`} />
                </button>
              </div>

              {!validHead && (
                <div className="text-xs text-red-400 bg-red-500/10 rounded p-2">
                  ⚠️ emb_dim ({cfg.emb_dim}) 必须能被 n_heads ({cfg.n_heads}) 整除
                </div>
              )}
              {validHead && (
                <div className="text-xs text-gray-500">每头维度: {headDim}</div>
              )}

              <button
                onClick={fetchAnalysis}
                disabled={loading || !validHead}
                className="w-full py-2 rounded bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white text-sm font-medium transition-colors"
              >
                {loading ? "分析中..." : "分析模型"}
              </button>
            </CardContent>
          </Card>
        </div>

        {/* Stats + architecture */}
        <div className="xl:col-span-2 space-y-4">
          {params && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-3">
                <div className="text-xs text-gray-500 mb-1">总参数量</div>
                <div className="text-2xl font-bold text-green-400">{params.total_M}M</div>
              </div>
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-3">
                <div className="text-xs text-gray-500 mb-1">推理显存</div>
                <div className="text-2xl font-bold text-blue-400">{memory?.inference_only_mb}MB</div>
              </div>
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-3">
                <div className="text-xs text-gray-500 mb-1">训练显存</div>
                <div className="text-2xl font-bold text-orange-400">{memory?.total_training_mb}MB</div>
              </div>
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-3">
                <div className="text-xs text-gray-500 mb-1">Transformer 层</div>
                <div className="text-2xl font-bold text-purple-400">{cfg.n_layers}</div>
              </div>
            </div>
          )}

          {/* Param breakdown */}
          {params && (
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-gray-300">参数分布</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {[
                    { label: "Token Embedding", value: params.token_embedding, color: "bg-purple-500" },
                    { label: "Position Embedding", value: params.position_embedding, color: "bg-purple-400" },
                    { label: `Transformer Blocks × ${params.num_blocks}`, value: params.all_blocks, color: "bg-blue-500" },
                    { label: "Final LayerNorm", value: params.final_layer_norm, color: "bg-green-500" },
                    { label: "Output Head", value: params.output_head, color: "bg-orange-500" },
                  ].map((item) => {
                    const pct = (item.value / params.total) * 100;
                    return (
                      <div key={item.label}>
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>{item.label}</span>
                          <span className="font-mono">{formatNum(item.value)} ({pct.toFixed(1)}%)</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full">
                          <div
                            className={`h-full rounded-full ${item.color}`}
                            style={{ width: `${Math.max(pct, 0.5)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Architecture layers */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">架构结构</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1.5">
              <LayerRow name="TokenEmbedding" type="embedding" shape={`[${cfg.vocab_size}, ${cfg.emb_dim}]`} params={cfg.vocab_size * cfg.emb_dim} />
              <LayerRow name="PositionEmbedding" type="embedding" shape={`[${cfg.context_length}, ${cfg.emb_dim}]`} params={cfg.context_length * cfg.emb_dim} />
              <LayerRow name="Dropout" type="dropout" shape={`[*, ${cfg.context_length}, ${cfg.emb_dim}]`} />

              {/* Transformer Blocks (show first, with toggle) */}
              <div
                className="cursor-pointer select-none"
                onClick={() => setExpandedBlock(!expandedBlock)}
              >
                <LayerRow
                  name={`▶ TransformerBlock × ${cfg.n_layers} (点击展开)`}
                  type="block"
                  params={params?.all_blocks}
                />
              </div>
              {expandedBlock && (
                <div className="ml-4 space-y-1 border-l border-gray-700 pl-3">
                  <LayerRow name="LayerNorm1" type="norm" shape={`[*, ${cfg.context_length}, ${cfg.emb_dim}]`} params={2 * cfg.emb_dim} />
                  <LayerRow name={`MultiHeadAttention (H=${cfg.n_heads}, head_dim=${headDim})`} type="attention" params={params ? params.per_block.attention_qkv + params.per_block.attention_out : undefined} />
                  <div className="ml-4 space-y-1 border-l border-gray-700 pl-3">
                    <LayerRow name={`W_query [${cfg.emb_dim} → ${cfg.emb_dim}]`} type="linear" params={cfg.emb_dim * cfg.emb_dim} />
                    <LayerRow name={`W_key   [${cfg.emb_dim} → ${cfg.emb_dim}]`} type="linear" params={cfg.emb_dim * cfg.emb_dim} />
                    <LayerRow name={`W_value [${cfg.emb_dim} → ${cfg.emb_dim}]`} type="linear" params={cfg.emb_dim * cfg.emb_dim} />
                    <LayerRow name={`out_proj [${cfg.emb_dim} → ${cfg.emb_dim}]`} type="linear" params={cfg.emb_dim * cfg.emb_dim + cfg.emb_dim} />
                  </div>
                  <LayerRow name="Residual +" type="residual" />
                  <LayerRow name="LayerNorm2" type="norm" shape={`[*, ${cfg.context_length}, ${cfg.emb_dim}]`} params={2 * cfg.emb_dim} />
                  <LayerRow name={`FeedForward [${cfg.emb_dim} → ${4 * cfg.emb_dim} → ${cfg.emb_dim}]`} type="ff" params={params?.per_block.feedforward} />
                  <div className="ml-4 space-y-1 border-l border-gray-700 pl-3">
                    <LayerRow name={`Linear [${cfg.emb_dim} → ${4 * cfg.emb_dim}]`} type="linear" />
                    <LayerRow name="GELU" type="activation" />
                    <LayerRow name={`Linear [${4 * cfg.emb_dim} → ${cfg.emb_dim}]`} type="linear" />
                  </div>
                  <LayerRow name="Residual +" type="residual" />
                </div>
              )}

              <LayerRow name="FinalLayerNorm" type="norm" shape={`[*, ${cfg.context_length}, ${cfg.emb_dim}]`} params={2 * cfg.emb_dim} />
              <LayerRow name={`OutputHead [${cfg.emb_dim} → ${cfg.vocab_size}]`} type="linear" shape={`[*, ${cfg.context_length}, ${cfg.vocab_size}]`} params={cfg.emb_dim * cfg.vocab_size} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
