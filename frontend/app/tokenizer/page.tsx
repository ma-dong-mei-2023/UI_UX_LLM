"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import TokenHighlighter from "@/components/tokenizer/TokenHighlighter";
import BpeAnimation from "@/components/tokenizer/BpeAnimation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from "lucide-react";

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
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play effect
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying && currentStep < steps.length - 1) {
      timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, 800);
    } else if (currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, steps.length]);

  // Encode results
  const [tiktokenEncoded, setTiktokenEncoded] = useState<any>(null);

  // Compare results
  const [compareResult, setCompareResult] = useState<any>(null);

  async function handleTrainBPE() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setIsPlaying(false);
    try {
      const result: any = await api.trainBPE(text, vocabSize);
      setSteps(result.steps);
      setCurrentStep(0); // Start at the beginning for animation
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
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Left Column: Config & Principle */}
            <div className="lg:col-span-1 space-y-4">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base text-gray-200">配置训练</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="w-full h-32 bg-gray-800 text-gray-200 text-sm rounded-lg p-3 border border-gray-700 focus:border-purple-500 focus:outline-none resize-none font-mono"
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
                      min={260}
                      max={1000}
                      step={10}
                      className="[&_[role=slider]]:bg-purple-500"
                    />
                    <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                      <span>260 (仅字节)</span><span>1000</span>
                    </div>
                  </div>
                  <Button
                    onClick={handleTrainBPE}
                    disabled={loading || !text.trim()}
                    className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors"
                  >
                    {loading ? "训练中..." : steps.length > 0 ? "重新训练 BPE" : "开始 BPE 训练"}
                  </Button>
                  {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
                </CardContent>
              </Card>

              {/* BPE Principle Card */}
              <Card className="bg-gray-900 border-gray-800 border-purple-900/30">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-purple-300 flex items-center gap-2">
                    <div className="w-1 h-4 bg-purple-500 rounded-full" />
                    BPE 训练原理
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-gray-400 space-y-2 leading-relaxed">
                  <p>字节对编码 (BPE) 是通过不断<strong>合并最高频相邻单元</strong>来构建词汇表的过程：</p>
                  <ul className="list-decimal list-inside space-y-1 ml-1">
                    <li><strong>字节初始化</strong>：将文本拆分为最小的可处理单位（字节）。</li>
                    <li><strong>统计频率</strong>：在当前序列中找出出现次数最频繁的相邻对。</li>
                    <li><strong>合并替换</strong>：将该最频繁对合并为一个新单元（如 'h' + 'e' → 'he'）。</li>
                    <li><strong>迭代更新</strong>：重复合并过程，直到达到预设的词表大小。</li>
                  </ul>
                  <p className="mt-2 text-purple-400/70 italic text-[10px]">
                    这使得分词器能自发学习文本模式，将常用词组缩减为单个 Token。
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Animation Viewer */}
            <div className="lg:col-span-2 space-y-4">
              {steps.length > 0 ? (
                <>
                  <Card className="bg-gray-900 border-gray-700">
                    <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                      <CardTitle className="text-base text-gray-200">
                        训练过程演示
                        <Badge variant="secondary" className="ml-2 bg-gray-800 text-gray-400 border-gray-700">
                          步数 {currentStep} / {steps.length - 1}
                        </Badge>
                      </CardTitle>
                      
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" size="icon" className="h-8 w-8 border-gray-700"
                          onClick={() => { setCurrentStep(0); setIsPlaying(false); }}
                          title="回到最初"
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="outline" size="icon" className="h-8 w-8 border-gray-700"
                          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                          disabled={currentStep === 0}
                        >
                          <SkipBack className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="outline" size="icon" className="h-8 w-8 bg-purple-600/20 border-purple-500/50 hover:bg-purple-600/40"
                          onClick={() => setIsPlaying(!isPlaying)}
                        >
                          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 fill-current" />}
                        </Button>
                        <Button 
                          variant="outline" size="icon" className="h-8 w-8 border-gray-700"
                          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
                          disabled={currentStep === steps.length - 1}
                        >
                          <SkipForward className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <BpeAnimation steps={steps} currentStep={currentStep} />
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card className="bg-gray-900 border-gray-800 border-dashed h-full flex flex-col items-center justify-center p-12 text-center">
                  <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
                    <Play className="h-8 w-8 text-gray-600 ml-1" />
                  </div>
                  <h3 className="text-gray-300 font-medium">准备好开始了吗？</h3>
                  <p className="text-gray-500 text-sm mt-2 max-w-xs">
                    点击左侧的“开始训练”按钮，我们将为您展示 BPE 是如何通过合并高频字符对来构建词汇表的。
                  </p>
                </Card>
              )}
            </div>
          </div>
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
