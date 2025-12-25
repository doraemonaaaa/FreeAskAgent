#!/usr/bin/env python3
"""
AgenticMemory 使用演示

展示如何在代码中使用 AgenticMemory 进行记忆管理和查询
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
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

def load_memory_component(name, class_name):
    """加载记忆组件"""
    component_path = PROJECT_ROOT / "agentflow" / "agentflow" / "models" / "memory" / f"{name}.py"

    spec = __import__('importlib.util').util.spec_from_file_location(name, component_path)
    module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def demo_memory_workflow():
    """演示记忆工作流程"""
    print("🎯 AgenticMemory 工作流程演示")
    print("=" * 50)

    try:
        # 加载 AgenticMemory
        AgenticMemory = load_memory_component('agentic_memory', 'AgenticMemory')

        # 创建记忆系统
        memory = AgenticMemory(
            enable_llm_features=True,
            llm_backend="litellm",
            llm_model="gpt-4o-mini",
            api_key=os.getenv('LITELLM_API_KEY'),
            api_base=os.getenv('LITELLM_API_BASE'),
            storage_dir="./demo_memory",
            evolution_threshold=3
        )

        print("✅ AgenticMemory 系统创建成功\n")

        # 演示 1: 添加记忆
        print("📝 添加记忆...")
        memories_to_add = [
            "时代广场中有盒马、永辉等大型超市，提供新鲜蔬果和日用品",
            "时代广场附近有星巴克咖啡店，环境舒适，适合工作和休息",
            "时代广场周边交通便利，有地铁站和多个公交站点",
            "时代广场是城市中心商业区，有很多餐厅和娱乐场所"
        ]

        memory_ids = []
        for content in memories_to_add:
            mem_id = memory.add_memory(content)
            memory_ids.append(mem_id)
            print(f"✅ 添加: {content[:30]}...")

        print(f"\n🎉 已添加 {len(memories_to_add)} 个记忆\n")

        # 演示 2: 查询记忆
        print("🔍 查询演示...")
        queries = [
            "时代广场周边有什么超市",
            "时代广场附近有咖啡店吗",
            "时代广场交通怎么样",
            "时代广场有什么娱乐设施"
        ]

        for query in queries:
            print(f"\n❓ 查询: {query}")
            results = memory.retrieve_memories(query, k=2)

            if results:
                print(f"🎯 找到 {len(results)} 个相关记忆:")
                for i, mem in enumerate(results, 1):
                    print(f"   {i}. {mem.content}")
            else:
                print("❌ 未找到相关记忆")

        # 演示 3: 智能分析
        print("\n🧠 记忆分析演示...")
        if memory_ids:
            mem = memory.get_memory(memory_ids[0])
            if mem:
                print(f"📄 记忆内容: {mem.content}")
                if hasattr(mem, 'keywords') and mem.keywords:
                    print(f"🔑 LLM 自动提取关键词: {mem.keywords}")
                if hasattr(mem, 'context') and mem.context:
                    print(f"📝 LLM 自动生成上下文: {mem.context}")
                if hasattr(mem, 'tags') and mem.tags:
                    print(f"🏷️ LLM 自动生成标签: {mem.tags}")

        # 演示 4: 统计信息
        print("\n📊 系统统计...")
        stats = memory.get_stats()
        print(f"   记忆总数: {stats.get('total_memories', 0)}")
        print(f"   记忆链接数: {stats.get('total_links', 0)}")
        print(f"   LLM功能: {'启用' if stats.get('llm_features_enabled') else '禁用'}")

        print("\n🎉 演示完成！您现在可以使用这个记忆系统了！")
        print("💡 提示: 运行 'python memory_cli.py' 启动交互式界面")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_simple_usage():
    """演示简单使用方法"""
    print("\n📚 简单使用示例")
    print("=" * 30)

    print("""
# 1. 导入 AgenticMemory
from agentflow.models.memory import AgenticMemory

# 2. 创建实例
memory = AgenticMemory(
    enable_llm_features=True,  # 启用LLM智能分析
    storage_dir="./my_memory"  # 指定存储目录
)

# 3. 添加记忆
memory.add_memory("时代广场中有盒马、永辉等超市")

# 4. 查询记忆
results = memory.retrieve_memories("时代广场周边有什么超市", k=3)

# 5. 使用结果
for mem in results:
    print(mem.content)

# 6. 查看统计
stats = memory.get_stats()
print(f"总记忆数: {stats['total_memories']}")
    """)

def main():
    """主函数"""
    print("🚀 AgenticMemory 使用演示")
    print("让 AI 记住一切，随时查询！")
    print()

    # 加载配置
    load_config()

    # 检查配置
    if not os.getenv('LITELLM_API_KEY'):
        print("❌ 未找到 API Key 配置")
        print("请检查 config.env 文件或设置环境变量")
        return

    print("✅ 配置检查通过\n")

    # 运行演示
    demo_memory_workflow()
    demo_simple_usage()

if __name__ == "__main__":
    main()
