#!/usr/bin/env python3
"""
AgenticMemory 交互式命令行工具

提供简单的命令行界面来添加和查询记忆，支持自然语言交互。
"""

import sys
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

def load_config():
    """加载配置文件"""
    config_file = PROJECT_ROOT / "agentflow" / "agentflow" / "models" / "memory" / "config.env"

    if config_file.exists():
        print("📄 加载配置文件...")
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        return True
    else:
        print("⚠️ 未找到配置文件，使用默认设置")
        return False

def load_memory_component(name, class_name):
    """加载记忆组件"""
    component_path = PROJECT_ROOT / "agentflow" / "agentflow" / "models" / "memory" / f"{name}.py"

    spec = __import__('importlib.util').util.spec_from_file_location(name, component_path)
    module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

class MemoryCLI:
    """AgenticMemory 命令行界面"""

    def __init__(self):
        self.memory_system = None
        self.commands = {
            'add': self.add_memory,
            'query': self.query_memory,
            'list': self.list_memories,
            'stats': self.show_stats,
            'clear': self.clear_memories,
            'help': self.show_help,
            'quit': self.quit_system
        }

    def initialize_memory_system(self):
        """初始化记忆系统"""
        try:
            print("🚀 初始化 AgenticMemory 系统...")

            # 加载 AgenticMemory
            AgenticMemory = load_memory_component('agentic_memory', 'AgenticMemory')

            # 创建实例
            self.memory_system = AgenticMemory(
                enable_llm_features=True,
                llm_backend="litellm",
                llm_model="gpt-4o-mini",
                api_key=os.getenv('LITELLM_API_KEY'),
                api_base=os.getenv('LITELLM_API_BASE'),
                storage_dir="./interactive_memory",
                evolution_threshold=5
            )

            print("✅ AgenticMemory 系统初始化成功！")
            print("💡 输入 'help' 查看可用命令\n")

        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            print("请检查配置文件和依赖项\n")
            return False

        return True

    def run(self):
        """运行命令行界面"""
        print("🤖 AgenticMemory 交互式工具")
        print("=" * 50)
        print("让 AI 记住一切，随时查询！")
        print("=" * 50)

        # 初始化系统
        if not self.initialize_memory_system():
            return

        # 主循环
        while True:
            try:
                user_input = input("\n📝 请输入命令 (help 查看帮助): ").strip()

                if not user_input:
                    continue

                # 解析命令
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # 执行命令
                if command in self.commands:
                    if command == 'quit':
                        break
                    self.commands[command](args)
                else:
                    # 如果不是命令，尝试作为记忆内容添加
                    if user_input.startswith(('add ', 'query ', 'list', 'stats', 'clear', 'help', 'quit')):
                        print("❌ 未知命令，请输入 'help' 查看帮助")
                    else:
                        # 直接添加为记忆
                        self.add_memory(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

    def add_memory(self, content):
        """添加记忆"""
        if not content:
            content = input("请输入要添加的记忆内容: ").strip()
            if not content:
                print("❌ 记忆内容不能为空")
                return

        try:
            print("🧠 正在分析并存储记忆...")

            # 添加记忆
            memory_id = self.memory_system.add_memory(content)

            # 显示结果
            memory = self.memory_system.get_memory(memory_id)
            print("✅ 记忆添加成功！")
            print(f"📄 内容: {memory.content}")

            if hasattr(memory, 'keywords') and memory.keywords:
                print(f"🔑 关键词: {', '.join(memory.keywords)}")

            if hasattr(memory, 'tags') and memory.tags:
                print(f"🏷️ 标签: {', '.join(memory.tags)}")

            if hasattr(memory, 'context') and memory.context:
                print(f"📝 上下文: {memory.context}")

        except Exception as e:
            print(f"❌ 添加记忆失败: {e}")

    def query_memory(self, query):
        """查询记忆"""
        if not query:
            query = input("请输入查询内容: ").strip()
            if not query:
                print("❌ 查询内容不能为空")
                return

        try:
            print("🔍 正在搜索相关记忆...")

            # 检索记忆
            results = self.memory_system.retrieve_memories(query, k=5)

            if not results:
                print("❌ 未找到相关记忆")
                return

            print(f"\n🎯 找到 {len(results)} 个相关记忆:\n")

            for i, memory in enumerate(results, 1):
                print(f"{i}. 📄 {memory.content}")

                if hasattr(memory, 'keywords') and memory.keywords:
                    print(f"   🔑 关键词: {', '.join(memory.keywords)}")

                if hasattr(memory, 'context') and memory.context:
                    print(f"   📝 上下文: {memory.context}")

                if hasattr(memory, 'tags') and memory.tags:
                    print(f"   🏷️ 标签: {', '.join(memory.tags)}")

                # 显示相似度分数（如果可用）
                if hasattr(memory, 'retrieval_count'):
                    print(f"   📊 检索次数: {memory.retrieval_count}")

                print()  # 空行分隔

        except Exception as e:
            print(f"❌ 查询失败: {e}")

    def list_memories(self, args=""):
        """列出所有记忆"""
        try:
            memories = self.memory_system.list_memories()

            if not memories:
                print("📝 当前没有任何记忆")
                return

            print(f"\n📚 共有 {len(memories)} 个记忆:\n")

            for i, memory in enumerate(memories, 1):
                print(f"{i}. 📄 {memory.content}")
                if hasattr(memory, 'tags') and memory.tags:
                    print(f"   🏷️ 标签: {', '.join(memory.tags)}")
                print()

        except Exception as e:
            print(f"❌ 列出记忆失败: {e}")

    def show_stats(self, args=""):
        """显示系统统计"""
        try:
            stats = self.memory_system.get_stats()

            print("\n📊 系统统计:"            print(f"   记忆总数: {stats.get('total_memories', 0)}")
            print(f"   记忆链接数: {stats.get('total_links', 0)}")
            print(f"   检索总次数: {stats.get('total_retrievals', 0)}")
            print(".1f"            print(f"   存储大小: {stats.get('storage_size_bytes', 0)} bytes")
            print(f"   LLM功能: {'启用' if stats.get('llm_features_enabled') else '禁用'}")

            if 'last_modified' in stats and stats['last_modified']:
                print(f"   最后修改: {stats['last_modified']}")

        except Exception as e:
            print(f"❌ 获取统计失败: {e}")

    def clear_memories(self, args=""):
        """清空所有记忆"""
        confirm = input("⚠️ 确定要清空所有记忆吗？(输入 'yes' 确认): ").strip().lower()

        if confirm == 'yes':
            try:
                # 重新初始化记忆系统（清空存储）
                self.memory_system = None

                # 删除存储目录
                import shutil
                storage_dir = Path("./interactive_memory")
                if storage_dir.exists():
                    shutil.rmtree(storage_dir)

                # 重新初始化
                self.initialize_memory_system()
                print("✅ 所有记忆已清空")

            except Exception as e:
                print(f"❌ 清空记忆失败: {e}")
        else:
            print("❌ 操作已取消")

    def show_help(self, args=""):
        """显示帮助信息"""
        print("""
🤖 AgenticMemory 交互式工具 - 帮助

📝 可用命令:

  add <内容>    - 添加新记忆
                 示例: add 时代广场中有盒马、永辉等超市

  query <问题>  - 查询相关记忆
                 示例: query 时代广场周边有什么超市

  list         - 列出所有记忆

  stats        - 显示系统统计

  clear        - 清空所有记忆

  help         - 显示此帮助

  quit         - 退出程序

💡 使用提示:

  • 直接输入文字（不带命令）会自动添加为记忆
  • 查询支持自然语言，系统会找到语义相关的记忆
  • 记忆会自动分析关键词和标签，便于后续检索
  • 支持中文内容，完全本地化

🎯 示例对话:

  用户: 在时代广场中有盒马、永辉等超市
  系统: ✅ 记忆添加成功！

  用户: query 时代广场周有什么超市
  系统: 🎯 找到 1 个相关记忆:
        1. 📄 在时代广场中有盒马、永辉等超市

🚀 享受智能记忆管理！
        """)

    def quit_system(self, args=""):
        """退出系统"""
        print("👋 感谢使用 AgenticMemory！再见！")
        return "quit"

def main():
    """主函数"""
    # 加载配置
    if not load_config():
        print("⚠️ 配置文件加载失败，使用默认设置")

    # 检查 API Key
    if not os.getenv('LITELLM_API_KEY'):
        print("❌ 未设置 LITELLM_API_KEY 环境变量")
        print("请在 config.env 文件中配置或手动设置:")
        print("export LITELLM_API_KEY='your-api-key'")
        return

    # 创建并运行 CLI
    cli = MemoryCLI()
    cli.run()

if __name__ == "__main__":
    main()

