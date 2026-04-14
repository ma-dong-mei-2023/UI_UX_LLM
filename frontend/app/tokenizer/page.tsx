"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import TokenHighlighter from "@/components/tokenizer/TokenHighlighter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const DEFAULT_TEXT = `The quick brown fox jumps over the lazy dog.
In the beginning was the Word, and the Word was with God.
Hello world! This is an example of byte pair encoding.`;

export default function TokenizerPage() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [vocabSize, setVocabSize] = useState(400);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // BPE train results
  const [steps, setSteps] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [finalVocabSize, setFinalVocabSize] = useState(0);

  // Encode results
  const [bpeEncoded, setBpeEncoded] = useState<any>(null);
  const [tiktokenEncoded, setTiktokenEncoded] = useState<any>(null);

  // Compare results
  const [compareResult, setCompareResult] = useState<any>(null);

  async function handleTrainBPE() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const result: any = await api.trainBPE(text, vocabSize);
      setSteps(result.steps);
      setCurrentStep(result.steps.length - 1);
      setFinalVocabSize(result.final_vocab_size);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCompare() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const [tikResult, cmpResult]: any = await Promise.all([
        api.encodeTiktoken(text),
        api.compareTokenizers(text, vocabSize),
      ]);
      setTiktokenEncoded(tikResult);
      setCompareResult(cmpResult);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const step = steps[currentStep];

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">✂️ 分词实验室</h1>
        <p className="text-gray-400 text-sm">
          第 2 章 · 可视化字节对编码 (BPE) 的训练过程，理解分词器是如何构建词汇表的。
        </p>
      </div>

      <Tabs defaultValue="bpe">
        <TabsList className="bg-gray-800 border border-gray-700">
          <TabsTrigger value="bpe" className="data-[state=active]:bg-purple-600">BPE 训练可视化</TabsTrigger>
          <TabsTrigger value="compare" className="data-[state=active]:bg-blue-600">分词器对比</TabsTrigger>
        </TabsList>

        {/* BPE Training Tab */}
        <TabsContent value="bpe" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Input */}
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-200">输入文本</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="w-full h-36 bg-gray-800 text-gray-200 text-sm rounded-lg p-3 border border-gray-700 focus:border-purple-500 focus:outline-none resize-none font-mono"
                  placeholder="输入要训练分词器的文本..."
                />
                <div>
                  <div className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>目标词汇表大小</span>
                    <span className="text-purple-400 font-mono">{vocabSize}</span>
                  </div>
                  <Slider
                    value={[vocabSize]}
                    onValueChange={(v) => setVocabSize(Array.isArray(v) ? v[0] : v)}
                    min={300}
                    max={2000}
                    step={50}
                    className="[&_[role=slider]]:bg-purple-500"
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>300</span><span>2000</span>
                  </div>
                </div>
                <button
                  onClick={handleTrainBPE}
                  disabled={loading || !text.trim()}
                  className="w-full py-2 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-medium transition-colors"
                >
                  {loading ? "训练中..." : "开始 BPE 训练"}
                </button>
                {error && <p className="text-red-400 text-sm">{error}</p>}
              </CardContent>
            </Card>

            {/* Stats */}
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-200">训练统计</CardTitle>
              </CardHeader>
              <CardContent>
                {steps.length > 0 ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-gray-800 rounded-lg p-3">
                        <div className="text-xs text-gray-500">总合并步数</div>
                        <div className="text-2xl font-bold text-purple-400">{steps.length}</div>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3">
                        <div className="text-xs text-gray-500">最终词汇表</div>
                        <div className="text-2xl font-bold text-blue-400">{finalVocabSize}</div>
                      </div>
                      {step && (
                        <>
                          <div className="bg-gray-800 rounded-lg p-3">
                            <div className="text-xs text-gray-500">当前 Token 数</div>
                            <div className="text-2xl font-bold text-green-400">{step.total_tokens}</div>
                          </div>
                          <div className="bg-gray-800 rounded-lg p-3">
                            <div className="text-xs text-gray-500">压缩比</div>
                            <div className="text-2xl font-bold text-orange-400">{step.compression_ratio}x</div>
                          </div>
                        </>
                      )}
                    </div>
                    {step && (
                      <div className="bg-gray-800 rounded-lg p-3 space-y-1">
                        <div className="text-xs text-gray-500">当前步骤合并</div>
                        <div className="flex items-center gap-2 font-mono text-sm">
                          <Badge variant="outline" className="text-purple-300 border-purple-500">
                            &quot;{step.merged_pair_str[0]}&quot;
                          </Badge>
                          <span className="text-gray-500">+</span>
                          <Badge variant="outline" className="text-purple-300 border-purple-500">
                            &quot;{step.merged_pair_str[1]}&quot;
                          </Badge>
                          <span className="text-gray-500">→</span>
                          <Badge className="bg-purple-600">
                            &quot;{step.new_token}&quot;
                          </Badge>
                          <span className="text-gray-500 text-xs">#{step.new_token_id}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-gray-600 text-sm text-center py-8">
                    训练后将在这里显示统计信息
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Step slider */}
          {steps.length > 0 && (
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-gray-200">
                  合并步骤浏览
                  <span className="text-gray-500 font-normal text-sm ml-2">
                    步骤 {currentStep + 1} / {steps.length}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Slider
                  value={[currentStep]}
                  onValueChange={(v) => setCurrentStep(Array.isArray(v) ? v[0] : v)}
                  min={0}
                  max={steps.length - 1}
                  step={1}
                  className="[&_[role=slider]]:bg-purple-500"
                />
                <div className="overflow-auto max-h-48">
                  <table className="w-full text-sm text-gray-300">
                    <thead>
                      <tr className="text-gray-500 text-xs border-b border-gray-700">
                        <th className="text-left pb-2">步骤</th>
                        <th className="text-left pb-2">合并对</th>
                        <th className="text-left pb-2">新 Token</th>
                        <th className="text-left pb-2">词汇表大小</th>
                        <th className="text-left pb-2">Token 总数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {steps.slice(Math.max(0, currentStep - 5), currentStep + 1).map((s) => (
                        <tr
                          key={s.step}
                          className={`border-b border-gray-800 font-mono text-xs ${
                            s.step === step?.step ? "bg-purple-500/10" : ""
                          }`}
                        >
                          <td className="py-1.5">{s.step}</td>
                          <td className="py-1.5">
                            &quot;{s.merged_pair_str[0]}&quot; + &quot;{s.merged_pair_str[1]}&quot;
                          </td>
                          <td className="py-1.5 text-purple-300">&quot;{s.new_token}&quot;</td>
                          <td className="py-1.5">{s.current_vocab_size}</td>
                          <td className="py-1.5">{s.total_tokens}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Compare Tab */}
        <TabsContent value="compare" className="space-y-4 mt-4">
          <Card className="bg-gray-900 border-gray-700">
            <CardContent className="pt-4 space-y-4">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full h-28 bg-gray-800 text-gray-200 text-sm rounded-lg p-3 border border-gray-700 focus:border-blue-500 focus:outline-none resize-none font-mono"
                placeholder="输入要分词的文本..."
              />
              <button
                onClick={handleCompare}
                disabled={loading || !text.trim()}
                className="py-2 px-6 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium transition-colors"
              >
                {loading ? "分词中..." : "对比分词结果"}
              </button>
              {error && <p className="text-red-400 text-sm">{error}</p>}
            </CardContent>
          </Card>

          {compareResult && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card className="bg-gray-900 border-blue-500/30">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center justify-between">
                    <span className="text-blue-300">tiktoken (GPT-2)</span>
                    <div className="flex gap-2">
                      <Badge className="bg-blue-500/20 text-blue-300">
                        {compareResult.tiktoken.num_tokens} tokens
                      </Badge>
                      <Badge className="bg-gray-700 text-gray-300">
                        词汇表 {compareResult.tiktoken.vocab_size.toLocaleString()}
                      </Badge>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <TokenHighlighter
                    tokens={compareResult.tiktoken.tokens.map((t: string, i: number) => ({
                      token: t,
                      id: compareResult.tiktoken.token_ids[i],
                    }))}
                  />
                  <div className="mt-2 text-xs text-gray-500">
                    压缩比: {compareResult.compression_tiktoken}x
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900 border-purple-500/30">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center justify-between">
                    <span className="text-purple-300">自定义 BPE</span>
                    <div className="flex gap-2">
                      <Badge className="bg-purple-500/20 text-purple-300">
                        {compareResult.custom_bpe.num_tokens} tokens
                      </Badge>
                      <Badge className="bg-gray-700 text-gray-300">
                        词汇表 {compareResult.custom_bpe.vocab_size}
                      </Badge>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <TokenHighlighter
                    tokens={compareResult.custom_bpe.tokens.map((t: string, i: number) => ({
                      token: t,
                      id: compareResult.custom_bpe.token_ids[i],
                    }))}
                  />
                  <div className="mt-2 text-xs text-gray-500">
                    压缩比: {compareResult.compression_bpe}x
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
