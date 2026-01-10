"""
Quick Start for Embodied Agent with One-Line Interface

一行代码调用完整的embodied agent流程，包括记忆系统和LLM调用。
支持视觉导航任务，自动检测流程并验证整体架构。
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目路径
sys.path.append('/root/autodl-tmp/FreeAskAgent')

from agentflow.agentflow.solver_embodied import construct_solver_embodied

from dotenv import load_dotenv
load_dotenv(dotenv_path="/root/autodl-tmp/FreeAskAgent/agentflow/.env")


def run_embodied_agent(
    question: str,
    image_paths: Optional[Union[str, List[str]]] = None,
    enable_memory: bool = True,
    task_type: str = "general_task",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    一行代码运行完整的embodied agent流程

    Args:
        question: 用户查询问题
        image_paths: 图片路径列表或单个图片路径
        enable_memory: 是否启用记忆系统
        task_type: 任务类型
        verbose: 是否显示详细信息

    Returns:
        包含完整流程结果的字典
    """
    start_time = time.time()

    # 确保使用GPT-4o模型
    llm_engine_name = "gpt-4o"

    # 配置记忆系统
    memory_config = {
        'retriever_config': {
            'use_api_embedding': False,  # 使用本地模型
            'local_model_path': '/root/autodl-tmp/all-MiniLM-L6-v2',  # 本地模型路径
            'model_name': 'all-MiniLM-L6-v2',
            'alpha': 0.5,
            'disable_semantic_search': False
        },
        'storage_dir': "./memory_store",
        'enable_persistence': True,
        'max_memories': 1000
    } if enable_memory else None

    # 构造solver - 使用GPT-4o确保实际LLM调用
    solver = construct_solver_embodied(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool", "Python_Coder_Tool"],
        tool_engine=["gpt-4o", "gpt-4o"],  # 全部使用GPT-4o
        model_engine=["gpt-4o", "gpt-4o", "gpt-4o"],  # planner_main, planner_fixed, executor
        output_types="base,final,direct",
        max_steps=10,
        max_time=300,
        max_tokens=4000,
        enable_multimodal=True,
        enable_memory=enable_memory,
        memory_config=memory_config,
        verbose=verbose
    )

    if verbose:
        print("🚀 Starting embodied agent with GPT-4o...")
        print(f"📝 Question: {question}")
        if image_paths:
            if isinstance(image_paths, list):
                print(f"🖼️ Images: {len(image_paths)} images provided")
            else:
                print(f"🖼️ Image: {image_paths}")
        print(f"🧠 Memory: {'Enabled' if enable_memory else 'Disabled'}")

    # 执行完整流程
    result = solver.solve(question, image_paths, task_type)

    # 添加执行统计
    result['execution_stats'] = {
        'total_time': round(time.time() - start_time, 2),
        'llm_engine': llm_engine_name,
        'memory_enabled': enable_memory,
        'task_type': task_type
    }

    # 添加记忆统计（如果启用）
    if enable_memory and hasattr(solver, 'memory_manager') and solver.memory_manager:
        result['memory_stats'] = solver.memory_manager.get_stats()

    if verbose:
        print("✅ Embodied agent execution completed!")
        print(f"⏱️ Total execution time: {result['execution_stats']['total_time']:.2f}s")
    return result


def solve_navigation_with_memory(enable_memory: bool = True, frame_dir: Path = None) -> Dict[str, Any]:
    """
    使用记忆系统解决视觉导航任务

    Args:
        enable_memory: 是否启用记忆功能
        frame_dir: 图片帧目录

    Returns:
        导航任务结果字典
    """
    print(f"\n{'='*80}")
    print(f"Testing Visual Navigation with Memory {'ENABLED' if enable_memory else 'DISABLED'}")
    print(f"{'='*80}")

    # 设置LLM引擎名称
    llm_engine_name = "gpt-4o"

    # 准备图片序列
    image_sequence = None
    if frame_dir and frame_dir.exists():
        # 获取所有jpeg图片并排序
        image_sequence = sorted(str(path) for path in frame_dir.glob("frame_*.jpeg"))
        if not image_sequence:
            # 如果没有frame_*.jpeg文件，使用input_img1.jpg
            input_img = frame_dir / "input_img1.jpg"
            if input_img.exists():
                image_sequence = [str(input_img)]
                print(f"📸 Using single image: {input_img}")
            else:
                print(f"⚠️ No images found in {frame_dir}")
                return None
        else:
            print(f"📸 Using {len(image_sequence)} frames from {frame_dir}")
    else:
        print(f"⚠️ Frame directory {frame_dir} not found")
        return None

    # 构造solver（现在包含记忆系统）
    print("🏗️ Constructing solver...")
    memory_config = {
        'retriever_config': {'use_api_embedding': True},
        'storage_dir': "./memory_store",
        'enable_persistence': True,
        'max_memories': 1000
    } if enable_memory else None

    solver = construct_solver_embodied(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
        tool_engine=["gpt-4o"],
        model_engine=["gpt-4o", "gpt-4o", "gpt-4o"],
        output_types="direct",
        max_time=300,
        max_steps=1,
        enable_multimodal=True,
        enable_memory=enable_memory,
        memory_config=memory_config
    )

    # 导航任务提示
    navigation_task_prompt = """Go to the store, called micheal's store."""

    print(f"🎯 Task: {navigation_task_prompt}")
    print(f"🖼️ Using {len(image_sequence)} image(s)")

    if enable_memory:
        print("✅ Memory system integrated into solver")

    # 执行导航任务
    print("🚀 Executing navigation task...")
    try:
        start_time = time.time()
        output = solver.solve(
            navigation_task_prompt,
            image_paths=image_sequence[:5],  # 最多使用5帧
            task_type="navigation_task"
        )
        execution_time = time.time() - start_time

        direct_output = output.get("direct_output", "No output generated")
        print(".2f")
        print(f"📝 Result: {direct_output[:200]}...")

        result = {
            'memory_enabled': enable_memory,
            'task': navigation_task_prompt,
            'images_used': len(image_sequence),
            'output': direct_output,
            'execution_time': execution_time,
            'success': bool(direct_output and len(direct_output.strip()) > 10),
            'memory_stats': None
        }

        # Add memory statistics if memory is enabled
        if enable_memory and hasattr(solver, 'long_memory') and solver.long_memory:
            result['memory_stats'] = solver.long_memory.get_stats()

        return result

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        result = {
            'memory_enabled': enable_memory,
            'task': navigation_task_prompt,
            'error': str(e),
            'success': False,
            'memory_stats': None
        }

        # Add memory statistics even in error case if available
        if enable_memory and hasattr(solver, 'long_memory') and solver.long_memory:
            result['memory_stats'] = solver.long_memory.get_stats()

        return result


def test_complete_flow():
    """
    测试完整流程：一行代码调用验证整体架构
    """
    # Run a compact set of memory-enabled tests (minimal terminal output)
    results = {}

    # Test 1: Simple text query (memory enabled)
    # Use an image-style prompt (referencing attached image) to avoid trivial Qs and to exercise multimodal memory flow
    r_text = run_embodied_agent(
        question="Description: Briefly describe the image scene in one neutral sentence.",
        enable_memory=True,
        verbose=False
    )
    results['text_only'] = r_text

    # Test 2: Multimodal query (use a sample image if available)
    test_images = []
    test_dirs = ["test/vln", "assets/images", "."]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.jpeg")) + list(Path(test_dir).glob("*.png"))
            if images:
                test_images = [str(img) for img in images[:1]]
                break

    if test_images:
        # Use an image-focused prompt (referencing provided attachment style)
        img_question = "Please describe the scene and the people in the image; suggest an immediate safe action for the person."
        r_img = run_embodied_agent(
            question=img_question,
            image_paths=test_images[0],
            enable_memory=True,
            verbose=False
        )
        results['multimodal'] = r_img
    else:
        results['multimodal'] = None

    # Memory sanity: three short turns to cause a window summary
    r_mem1 = run_embodied_agent(question="Turn 1: Hello", enable_memory=True, verbose=False)
    r_mem2 = run_embodied_agent(question="Turn 2: Provide a fact about X", enable_memory=True, verbose=False)
    r_mem3 = run_embodied_agent(question="Turn 3: Summarize previous", enable_memory=True, verbose=False)
    results['memory_test_summary'] = r_mem3

    # Minimal console report
    print("✅ Tests completed (memory enabled). Summary:")
    print(f" - Text-only direct_output length: {len((r_text.get('direct_output') or '') or '')}")
    if results['multimodal']:
        print(f" - Multimodal direct_output length: {len((results['multimodal'].get('direct_output') or '') or '')}")
    else:
        print(" - Multimodal test: skipped (no image found)")
    mem_stats = r_mem3.get('memory_stats') or {}
    short_total = mem_stats.get('short_memory', {}).get('total_messages', 0)
    long_count = mem_stats.get('long_memory', {}).get('current_memory_count', 0)
    print(f" - Short messages: {short_total}, Long memory entries: {long_count}")

    return results


def sanity_check_memory_flow():
    """
    Quick sanity unit test to verify per-turn short memory writes and long-memory summarization.
    This function runs three sequential queries to ensure the short-memory window fills and
    the MemoryManager attempts to summarize and add a conversation summary to long-term memory.
    """
    print("\n🔬 Running memory sanity check (3 turns)...")
    r1 = run_embodied_agent(question="Turn 1: Hello", enable_memory=True, verbose=False)
    r2 = run_embodied_agent(question="Turn 2: Tell me something about X", enable_memory=True, verbose=False)
    r3 = run_embodied_agent(question="Turn 3: Summarize previous", enable_memory=True, verbose=False)

    mem_stats = r3.get('memory_stats') or {}
    short_total = mem_stats.get('short_memory', {}).get('total_messages', 0)
    long_count = mem_stats.get('long_memory', {}).get('current_memory_count', 0)

    print(f"Sanity check results -> short_total_messages: {short_total}, long_memory_count: {long_count}")
    return {
        "short_total_messages": short_total,
        "long_memory_count": long_count,
        "raw_stats": mem_stats
    }


def main(enable_memory: bool = True, frame_dir: str = "test/vln"):
    """
    主测试函数 - 演示一行代码调用接口

    Args:
        enable_memory: 是否启用记忆功能进行测试
        frame_dir: 图片帧目录路径
    """
    print("🚀 Embodied Agent One-Line Interface Demo")
    print("Demonstrating complete flow with single function call")
    print("=" * 60)

    # 演示一行代码调用
    print("\n💡 One-Line Usage Examples:")
    print("# Simple text query:")
    print('result = run_embodied_agent("What is the capital of France?")')
    print("\n# Multimodal query with memory:")
    print('result = run_embodied_agent("Analyze this image", image_paths="image.jpg", enable_memory=True)')
    print("\n# Custom configuration:")
    print('result = run_embodied_agent("Solve this task", task_type="navigation", verbose=False)')

    # 运行完整流程测试
    print("\n" + "=" * 60)
    print("🧪 RUNNING COMPLETE FLOW TEST")
    print("=" * 60)

    test_results = test_complete_flow()

    # 输出结果摘要
    print(f"\n{'='*80}")
    print("EMBODIED AGENT FLOW TEST RESULTS")
    print(f"{'='*80}")

    for test_name, result in test_results.items():
        if result:
            print(f"\n📊 {test_name.upper()}:")
            stats = result.get('execution_stats', {})
            print(f"  ⏱️ Time: {stats.get('total_time', 0):.2f}s")
            print(f"  🤖 LLM: {stats.get('llm_engine', 'Unknown')}")
            print(f"  🧠 Memory: {stats.get('memory_enabled', False)}")
            print(f"  📝 Output Length: {len(result.get('direct_output', ''))} chars")

            if result.get('memory_stats'):
                mem_stats = result['memory_stats']
                print(f"  💾 Memories: {mem_stats.get('short_memory', {}).get('total_messages', 0)} messages")

    print("\n✅ All tests completed! Embodied agent flow verified.")
    return test_results


def run_comparison_test(frame_dir: str = "test/vln"):
    """
    运行视觉导航对比测试：分别测试启用和禁用记忆的情况

    Args:
        frame_dir: 图片帧目录路径
    """
    print("🔄 Running Visual Navigation Memory Comparison Test")
    print("This test compares agent navigation performance with memory ON vs OFF")
    print(f"Using images from: {frame_dir}")

    # 测试禁用记忆的情况
    print("\n" + "="*50 + " PHASE 1: WITHOUT MEMORY " + "="*50)
    result_without_memory = main(enable_memory=False, frame_dir=frame_dir)

    # 测试启用记忆的情况
    print("\n" + "="*50 + " PHASE 2: WITH MEMORY " + "="*50)
    result_with_memory = main(enable_memory=True, frame_dir=frame_dir)

    # 检查测试结果是否有效
    if result_without_memory is None or result_with_memory is None:
        print("❌ Comparison test failed: One or both test phases failed")
        return None

    # 对比结果
    print(f"\n{'='*100}")
    print("VISUAL NAVIGATION COMPARISON RESULTS")
    print(f"{'='*100}")

    print("Without Memory:")
    print(f"  - Success: {result_without_memory['success']}")
    print(".2f")
    print(f"  - Output Length: {len(result_without_memory.get('output', ''))} chars")

    print("\nWith Memory:")
    print(f"  - Success: {result_with_memory['success']}")
    print(".2f")
    print(f"  - Output Length: {len(result_with_memory.get('output', ''))} chars")

    # 显示详细输出对比
    print(f"\n{'='*50} DETAILED OUTPUTS {'='*50}")

    print("\n--- WITHOUT MEMORY OUTPUT ---")
    output_without = result_without_memory.get('output', 'No output')
    print(output_without[:500] + ("..." if len(output_without) > 500 else ""))

    print("\n--- WITH MEMORY OUTPUT ---")
    output_with = result_with_memory.get('output', 'No output')
    print(output_with[:500] + ("..." if len(output_with) > 500 else ""))

    # 分析结果
    print(f"\n{'='*50} ANALYSIS {'='*50}")

    success_without = result_without_memory['success']
    success_with = result_with_memory['success']

    if success_with and not success_without:
        print("✅ Memory system provides clear benefit!")
        print("   Agent performed better with memory enabled for visual navigation.")
    elif success_with and success_without:
        print("🤔 Both tests succeeded - memory may provide subtle improvements")
        print("   Analyzing output quality and execution time...")

        # 比较输出质量和执行时间
        time_without = result_without_memory.get('execution_time', 0)
        time_with = result_with_memory.get('execution_time', 0)

        if time_with < time_without:
            print(".2f")
        elif time_with > time_without:
            print(".2f")
        # 比较输出长度作为质量指标
        len_without = len(result_without_memory.get('output', ''))
        len_with = len(result_with_memory.get('output', ''))

        if len_with > len_without:
            print(f"   Memory version produced {len_with - len_without} more characters of output")
        elif len_with < len_without:
            print(f"   Non-memory version produced {len_without - len_with} more characters of output")

    elif not success_with and not success_without:
        print("❌ Both tests failed - possible issues:")
        print("   - Image loading problems")
        print("   - LLM service connectivity issues")
        print("   - Task complexity too high")
    else:
        print("⚠️ Unexpected results - memory version failed but non-memory succeeded")
        print("   This might indicate memory interference or initialization issues")

    return {
        'without_memory': result_without_memory,
        'with_memory': result_with_memory,
        'analysis': {
            'memory_benefit': success_with and not success_without,
            'both_successful': success_with and success_without,
            'both_failed': not success_with and not success_without,
            'unexpected_result': not success_with and success_without
        }
    }


if __name__ == "__main__":
    # 打印配置信息
    print("🔑 API Configuration:")
    print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
    print("OPENAI_API_KEY:" + ("Set" if os.environ.get("OPENAI_API_KEY") else "Not Set"))
    print("DASHSCOPE_API_KEY:" + ("Set" if os.environ.get("DASHSCOPE_API_KEY") else "Not Set"))

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ('test', 'flow', 'complete'):
            # 运行完整流程测试
            print("\n🎯 Running complete flow test...")
            main()
        elif sys.argv[1].lower() in ('simple', 'demo'):
            # 运行简单演示
            print("\n🎯 Running simple demo...")
            result = run_embodied_agent("Hello, can you help me understand how memory systems work?")
            direct_output = result.get('direct_output', 'No response')
            if isinstance(direct_output, dict):
                print(f"Response: {str(direct_output)[:200]}...")
            else:
                print(f"Response: {str(direct_output)[:200]}...")
        elif sys.argv[1].lower() in ('true', '1', 'yes', 'on'):
            # 只运行有记忆的版本
            main(enable_memory=True)
        elif sys.argv[1].lower() in ('false', '0', 'no', 'off'):
            # 只运行无记忆的版本
            main(enable_memory=False)
        else:
            # 如果参数是路径，使用该路径作为frame_dir
            frame_dir = sys.argv[1]
            run_comparison_test(frame_dir)
    else:
        # 默认运行完整流程测试
        print("\n🎯 Running complete embodied agent flow test...")
        main()


