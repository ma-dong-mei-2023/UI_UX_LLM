"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const modules = [
  { href: "/", label: "首页", icon: "🏠", desc: "学习路径总览" },
  { href: "/tokenizer", label: "分词实验室", icon: "✂️", desc: "Ch2: BPE 可视化" },
  { href: "/attention", label: "注意力可视化", icon: "👁️", desc: "Ch3: 注意力机制" },
  { href: "/architecture", label: "架构构建器", icon: "🏗️", desc: "Ch4: GPT 模型结构" },
  { href: "/training", label: "训练仪表盘", icon: "📈", desc: "Ch5: 预训练" },
  { href: "/chat", label: "对话界面", icon: "💬", desc: "Ch7: 指令微调" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 min-h-screen bg-gray-900 text-white flex flex-col border-r border-gray-700">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold text-blue-400">LLM 从零开始</h1>
        <p className="text-xs text-gray-400 mt-1">交互式学习平台</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {modules.map((mod) => {
          const active = pathname === mod.href;
          return (
            <Link
              key={mod.href}
              href={mod.href}
              className={`flex items-start gap-3 px-3 py-2.5 rounded-lg transition-colors group ${
                active
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:bg-gray-800 hover:text-white"
              }`}
            >
              <span className="text-xl flex-shrink-0 mt-0.5">{mod.icon}</span>
              <div>
                <div className="font-medium text-sm">{mod.label}</div>
                <div className={`text-xs ${active ? "text-blue-200" : "text-gray-500 group-hover:text-gray-400"}`}>
                  {mod.desc}
                </div>
              </div>
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-gray-700 text-xs text-gray-500">
        <p>基于 Sebastian Raschka</p>
        <p>《Build a LLM From Scratch》</p>
      </div>
    </aside>
  );
}
