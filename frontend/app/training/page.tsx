"use client";
import { useState, useCallback, useRef } from "react";
import { api, WS_BASE } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type { TrainingMetric } from "@/lib/types";

const PRESETS = ["tiny", "small", "gpt2-124m", "gpt2-355m"];
const PRESET_LABELS: Record<string, string> = {
  tiny: "Tiny (~300K)", small: "Small (~3M)", "gpt2-124m": "GPT-2 124M", "gpt2-355m": "GPT-2 355M"
};

export default function TrainingPage() {
  const [preset, setPreset] = useState("tiny");
  const [lr, setLr] = useState(5e-4);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(2);
  const [evalFreq, setEvalFreq] = useState(5);
  const [useWarmup, setUseWarmup] = useState(false);
  const [startContext, setStartContext] = useState("Every effort moves");

  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "cancelled" | "error">("idle");
  const [metrics, setMetrics] = useState<any[]>([]);
  const [samples, setSamples] = useState<{ epoch: number; text: string }[]>([]);
  const [error, setError] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  const chartData = metrics
    .filter((m) => m.type === "eval")
    .map((m) => ({
      step: m.global_step,
      train: m.train_loss,
      val: m.val_loss,
      lr: m.learning_rate,
    }));

  function connectWebSocket(rid: string) {
    const ws = new WebSocket(`${WS_BASE}/ws/training/${rid}`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      setMetrics((prev) => [...prev, msg]);
      if (msg.type === "epoch_end") {
        setSamples((prev) => [...prev, { epoch: msg.epoch, text: msg.sample_text }]);
      }
      if (["completed", "cancelled", "error"].includes(msg.type)) {
        setStatus(msg.type as any);
        if (msg.message) setError(msg.message);
      }
    };
    ws.onerror = () => setError("WebSocket 连接失败");
  }

  async function handleStart() {
    setMetrics([]); setSamples([]); setError(""); setStatus("running");
    try {
      const result: any = await api.startTraining({
        model_preset: preset,
        learning_rate: lr,
        num_epochs: epochs,
        batch_size: batchSize,
        eval_freq: evalFreq,
        use_warmup: useWarmup,
        start_context: startContext,
      });
      setRunId(result.run_id);
      connectWebSocket(result.run_id);
    } catch (e: any) {
      setError(e.message);
      setStatus("error");
    }
  }

  async function handleCancel() {
    if (!runId) return;
    try {
      await api.cancelTraining(runId);
    } catch (e: any) {
      setError(e.message);
    }
    wsRef.current?.close();
    setStatus("cancelled");
  }

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">📈 训练仪表盘</h1>
        <p className="text-gray-400 text-sm">
          第 5 章 · 在真实文本上从零预训练 GPT 模型，实时观察损失下降和生成质量提升。
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Config */}
        <div className="space-y-4">
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">模型预设</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => setPreset(p)}
                  disabled={status === "running"}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition-colors ${
                    preset === p ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                  } disabled:opacity-50`}
                >
                  {PRESET_LABELS[p]}
                </button>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">训练参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { label: "Epochs", value: epochs, set: setEpochs, min: 1, max: 20, step: 1 },
                { label: "Batch Size", value: batchSize, set: setBatchSize, min: 1, max: 8, step: 1 },
                { label: "评估频率 (步)", value: evalFreq, set: setEvalFreq, min: 1, max: 50, step: 1 },
              ].map(({ label, value, set, min, max, step }) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-gray-400">{label}</span>
                    <span className="text-orange-400 font-mono">{value}</span>
                  </div>
                  <Slider value={[value]} onValueChange={(v) => set(Array.isArray(v) ? v[0] : v)}
                    min={min} max={max} step={step}
                    disabled={status === "running"}
                    className="[&_[role=slider]]:bg-orange-500" />
                </div>
              ))}

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-400">学习率</span>
                  <span className="text-orange-400 font-mono">{lr.toExponential(1)}</span>
                </div>
                <Slider value={[Math.log10(lr) * -1]} onValueChange={(v) => setLr(Math.pow(10, -(Array.isArray(v) ? v[0] : v)))}
                  min={2} max={5} step={0.5}
                  disabled={status === "running"}
                  className="[&_[role=slider]]:bg-orange-500" />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">学习率预热</span>
                <button
                  onClick={() => setUseWarmup(!useWarmup)}
                  disabled={status === "running"}
                  className={`w-10 h-5 rounded-full transition-colors ${useWarmup ? "bg-orange-600" : "bg-gray-700"}`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white mx-0.5 transition-transform ${useWarmup ? "translate-x-5" : ""}`} />
                </button>
              </div>

              <div>
                <label className="text-xs text-gray-400 block mb-1">生成起始文本</label>
                <input
                  value={startContext}
                  onChange={(e) => setStartContext(e.target.value)}
                  disabled={status === "running"}
                  className="w-full bg-gray-800 text-gray-200 text-sm rounded px-3 py-1.5 border border-gray-700 focus:border-orange-500 focus:outline-none"
                />
              </div>

              {status === "idle" || status === "completed" || status === "cancelled" || status === "error" ? (
                <button
                  onClick={handleStart}
                  className="w-full py-2 rounded bg-orange-600 hover:bg-orange-500 text-white font-medium text-sm"
                >
                  开始训练
                </button>
              ) : (
                <button
                  onClick={handleCancel}
                  className="w-full py-2 rounded bg-red-700 hover:bg-red-600 text-white font-medium text-sm"
                >
                  取消训练
                </button>
              )}

              {error && <p className="text-red-400 text-xs">{error}</p>}

              {runId && (
                <div className="text-xs text-gray-600 flex items-center gap-2">
                  <span>Run ID:</span>
                  <span className="font-mono text-gray-400">{runId}</span>
                  <Badge className={
                    status === "running" ? "bg-yellow-500/20 text-yellow-300" :
                    status === "completed" ? "bg-green-500/20 text-green-300" :
                    "bg-red-500/20 text-red-300"
                  }>{status}</Badge>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="xl:col-span-2 space-y-4">
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">
                损失曲线
                {status === "running" && (
                  <span className="ml-2 inline-block w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="step" stroke="#6B7280" tick={{ fontSize: 11 }} label={{ value: "步骤", position: "insideBottom", fill: "#6B7280", fontSize: 11 }} />
                    <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: "8px" }}
                      labelStyle={{ color: "#9CA3AF" }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#9CA3AF" }} />
                    <Line type="monotone" dataKey="train" stroke="#F97316" strokeWidth={2} dot={false} name="训练损失" />
                    <Line type="monotone" dataKey="val" stroke="#60A5FA" strokeWidth={2} dot={false} name="验证损失" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
                  {status === "running" ? "等待第一个评估点..." : "开始训练后显示损失曲线"}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Generated samples */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">生成文本样本</CardTitle>
            </CardHeader>
            <CardContent>
              {samples.length > 0 ? (
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {samples.map((s, i) => (
                    <div key={i} className="bg-gray-800 rounded-lg p-3">
                      <div className="text-xs text-gray-500 mb-1">Epoch {s.epoch}</div>
                      <p className="text-sm text-gray-300 font-mono leading-relaxed">{s.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-600 text-sm text-center py-6">
                  每个 epoch 结束后显示生成的文本样本
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
