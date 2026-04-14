"use client";
import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  role: "user" | "assistant";
  instruction: string;
  input?: string;
  response: string;
}

export default function ChatPage() {
  const [instruction, setInstruction] = useState("");
  const [inputText, setInputText] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(128);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState("");
  const [samples, setSamples] = useState<any[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    loadSamples();
  }, []);

  async function loadSamples() {
    setSamplesLoading(true);
    try {
      const result: any = await api.getInstructionSamples(10, 0);
      setSamples(result.data || []);
    } catch { }
    setSamplesLoading(false);
  }

  async function updatePreview() {
    if (!instruction.trim()) { setPreview(""); return; }
    try {
      const result: any = await api.formatPreview(instruction, inputText);
      setPreview(result.formatted || "");
    } catch { }
  }

  async function handleSend() {
    if (!instruction.trim()) return;
    setLoading(true);
    const msg: Message = { role: "user", instruction, input: inputText, response: "" };
    setMessages((prev) => [...prev, msg]);

    try {
      // In demo mode without a loaded model, show the prompt
      const result: any = await api.formatPreview(instruction, inputText);
      const response = "[模型未加载] 完成训练或加载预训练权重后，这里将显示真实的模型响应。\n\n格式化的提示词已生成：\n\n" + result.formatted;
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { ...updated[updated.length - 1], response };
        return updated;
      });
    } catch (e: any) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { ...updated[updated.length - 1], response: `错误: ${e.message}` };
        return updated;
      });
    }
    setInstruction(""); setInputText("");
    setLoading(false);
  }

  function useSample(sample: any) {
    setInstruction(sample.instruction || "");
    setInputText(sample.input || "");
    updatePreview();
  }

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">💬 对话界面</h1>
        <p className="text-gray-400 text-sm">
          第 7 章 · 与指令微调模型对话。完成训练后可在此与模型交互。
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: Config + Samples */}
        <div className="space-y-4">
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">生成参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { label: "Temperature", value: temperature, set: setTemperature, min: 0, max: 2, step: 0.1, color: "pink" },
                { label: "Top-K", value: topK, set: setTopK, min: 1, max: 100, step: 1, color: "pink" },
                { label: "最大 Token 数", value: maxTokens, set: setMaxTokens, min: 32, max: 512, step: 32, color: "pink" },
              ].map(({ label, value, set, min, max, step }) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-gray-400">{label}</span>
                    <span className="text-pink-400 font-mono">{value}</span>
                  </div>
                  <Slider value={[value]} onValueChange={(v) => set(Array.isArray(v) ? v[0] : v)}
                    min={min} max={max} step={step}
                    className="[&_[role=slider]]:bg-pink-500" />
                </div>
              ))}
              <div className="text-xs text-gray-500 mt-2 space-y-1">
                <p>• <strong className="text-gray-400">Temperature</strong>: 越高输出越随机</p>
                <p>• <strong className="text-gray-400">Top-K</strong>: 每步只从最高概率的 K 个词中采样</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-gray-300">示例指令</CardTitle>
            </CardHeader>
            <CardContent>
              {samplesLoading ? (
                <div className="text-gray-600 text-sm text-center py-4">加载中...</div>
              ) : (
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {samples.map((s, i) => (
                      <button
                        key={i}
                        onClick={() => useSample(s)}
                        className="w-full text-left p-2 rounded bg-gray-800 hover:bg-gray-700 text-xs text-gray-300 transition-colors"
                      >
                        <div className="font-medium text-gray-200 truncate">{s.instruction}</div>
                        {s.input && <div className="text-gray-500 truncate mt-0.5">{s.input}</div>}
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right: Chat + Preview */}
        <div className="xl:col-span-2 space-y-4">
          <Tabs defaultValue="chat">
            <TabsList className="bg-gray-800 border border-gray-700">
              <TabsTrigger value="chat" className="data-[state=active]:bg-pink-600">对话</TabsTrigger>
              <TabsTrigger value="preview" className="data-[state=active]:bg-purple-600">指令格式预览</TabsTrigger>
            </TabsList>

            <TabsContent value="chat" className="space-y-4 mt-4">
              {/* Messages */}
              <Card className="bg-gray-900 border-gray-700 h-96 flex flex-col">
                <CardContent className="flex-1 overflow-auto p-4 space-y-4">
                  {messages.length === 0 && (
                    <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                      输入指令开始对话，或从左侧选择示例指令
                    </div>
                  )}
                  {messages.map((msg, i) => (
                    <div key={i} className="space-y-2">
                      <div className="bg-pink-500/10 border border-pink-500/20 rounded-lg p-3">
                        <div className="text-xs text-pink-400 mb-1">指令</div>
                        <div className="text-sm text-gray-200">{msg.instruction}</div>
                        {msg.input && (
                          <>
                            <div className="text-xs text-pink-400 mb-1 mt-2">输入</div>
                            <div className="text-sm text-gray-300">{msg.input}</div>
                          </>
                        )}
                      </div>
                      {msg.response && (
                        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 ml-4">
                          <div className="text-xs text-gray-500 mb-1">模型响应</div>
                          <div className="text-sm text-gray-200 whitespace-pre-wrap font-mono">{msg.response}</div>
                        </div>
                      )}
                    </div>
                  ))}
                  {loading && (
                    <div className="flex items-center gap-2 text-gray-500 text-sm ml-4">
                      <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce [animation-delay:0.1s]" />
                      <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                    </div>
                  )}
                  <div ref={bottomRef} />
                </CardContent>
              </Card>

              {/* Input */}
              <Card className="bg-gray-900 border-gray-700">
                <CardContent className="pt-4 space-y-3">
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">指令 (Instruction)</label>
                    <textarea
                      value={instruction}
                      onChange={(e) => setInstruction(e.target.value)}
                      onBlur={updatePreview}
                      placeholder="描述任务，例如：将下面的句子翻译成英文"
                      className="w-full h-20 bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:border-pink-500 focus:outline-none resize-none"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">输入 (Input, 可选)</label>
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="任务的具体输入内容（可选）"
                      className="w-full h-16 bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:border-pink-500 focus:outline-none resize-none"
                    />
                  </div>
                  <button
                    onClick={handleSend}
                    disabled={loading || !instruction.trim()}
                    className="w-full py-2 rounded bg-pink-600 hover:bg-pink-500 disabled:opacity-50 text-white font-medium text-sm"
                  >
                    {loading ? "生成中..." : "发送"}
                  </button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="preview" className="mt-4">
              <Card className="bg-gray-900 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-gray-300">Alpaca 指令格式</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-gray-500 mb-3">
                    第 7 章使用 Alpaca 格式对 GPT-2 进行指令微调，模型接受以下格式的提示词：
                  </p>
                  <pre className="bg-gray-800 rounded-lg p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto max-h-64">
                    {preview || `Below is an instruction that describes a task...

### Instruction:
[你的指令]

### Input:
[可选输入]

### Response:`}
                  </pre>
                  <button
                    onClick={updatePreview}
                    className="mt-3 py-1.5 px-4 rounded bg-purple-700 hover:bg-purple-600 text-white text-sm"
                  >
                    刷新预览
                  </button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
