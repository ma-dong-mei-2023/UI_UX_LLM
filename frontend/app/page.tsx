import Link from "next/link";

const modules = [
  {
    href: "/tokenizer",
    icon: "✂️",
    title: "分词实验室",
    chapter: "第 2 章",
    desc: "可视化 BPE 字节对编码的每一步合并过程，对比自定义分词器与 tiktoken GPT-2 的差异。",
    topics: ["字节对编码 (BPE)", "词汇表增长", "tiktoken 对比"],
    color: "from-purple-500/20 to-purple-500/5 border-purple-500/30",
    badge: "bg-purple-500/20 text-purple-300",
  },
  {
    href: "/attention",
    icon: "👁️",
    title: "注意力可视化",
    chapter: "第 3 章",
    desc: "交互式注意力权重热力图，理解自注意力、因果掩码和多头注意力的工作原理。",
    topics: ["自注意力", "因果掩码", "多头注意力"],
    color: "from-blue-500/20 to-blue-500/5 border-blue-500/30",
    badge: "bg-blue-500/20 text-blue-300",
  },
  {
    href: "/architecture",
    icon: "🏗️",
    title: "架构构建器",
    chapter: "第 4 章",
    desc: "通过滑块配置 GPT 模型的每个参数，实时查看参数量、显存占用和架构图。",
    topics: ["层归一化", "前馈网络", "参数计算"],
    color: "from-green-500/20 to-green-500/5 border-green-500/30",
    badge: "bg-green-500/20 text-green-300",
  },
  {
    href: "/training",
    icon: "📈",
    title: "训练仪表盘",
    chapter: "第 5 章",
    desc: "在真实文本上预训练 GPT 模型，实时查看损失曲线，观察每个 epoch 的文本生成质量。",
    topics: ["预训练循环", "损失监控", "文本生成"],
    color: "from-orange-500/20 to-orange-500/5 border-orange-500/30",
    badge: "bg-orange-500/20 text-orange-300",
  },
  {
    href: "/chat",
    icon: "💬",
    title: "对话界面",
    chapter: "第 7 章",
    desc: "与指令微调后的模型对话，调整生成参数，观察温度和 top-k 对输出的影响。",
    topics: ["指令微调", "温度采样", "流式生成"],
    color: "from-pink-500/20 to-pink-500/5 border-pink-500/30",
    badge: "bg-pink-500/20 text-pink-300",
  },
];

export default function HomePage() {
  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-3">大语言模型从零开始</h1>
        <p className="text-gray-400 text-lg max-w-2xl">
          基于 Sebastian Raschka《Build a Large Language Model From Scratch》的交互式学习平台。
          通过可视化实验真正理解 LLM 的每一个核心组件。
        </p>
      </div>

      <div className="mb-8">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">学习路径</h2>
        <div className="flex items-center gap-2 text-sm text-gray-400 mb-6 flex-wrap">
          {modules.map((m, i) => (
            <div key={m.href} className="flex items-center gap-2">
              <span className="text-gray-600">{m.icon}</span>
              <span>{m.title}</span>
              {i < modules.length - 1 && <span className="text-gray-700">→</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {modules.map((mod) => (
          <Link
            key={mod.href}
            href={mod.href}
            className={`group block p-5 rounded-xl border bg-gradient-to-br ${mod.color} hover:scale-[1.02] transition-transform`}
          >
            <div className="flex items-start justify-between mb-3">
              <span className="text-3xl">{mod.icon}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${mod.badge}`}>
                {mod.chapter}
              </span>
            </div>
            <h3 className="text-white font-semibold text-lg mb-1 group-hover:text-blue-300 transition-colors">
              {mod.title}
            </h3>
            <p className="text-gray-400 text-sm mb-3 leading-relaxed">{mod.desc}</p>
            <div className="flex flex-wrap gap-1.5">
              {mod.topics.map((t) => (
                <span key={t} className="text-xs bg-white/5 text-gray-400 px-2 py-0.5 rounded">
                  {t}
                </span>
              ))}
            </div>
          </Link>
        ))}
      </div>

      <div className="mt-10 p-5 rounded-xl border border-gray-800 bg-gray-900/50">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">📚 配套书籍</h3>
        <p className="text-gray-500 text-sm">
          Sebastian Raschka《Build a Large Language Model (From Scratch)》· Manning Publications 2024 ·
          ISBN: 9781633437166
        </p>
      </div>
    </div>
  );
}
