"""
Quick Start Test for Embodied Agent Memory System with Visual Navigation

测试脚本验证记忆系统在视觉导航任务中的作用。
比较有记忆版本和无记忆版本在使用VLN图片下的性能差异。
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append('/root/autodl-tmp/FreeAskAgent')

from agentflow.agentflow.solver_embodied import construct_solver_embodied

from dotenv import load_dotenv
load_dotenv(dotenv_path="agentflow/.env")


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


def main(enable_memory: bool = True, frame_dir: str = "test/vln"):
    """
    主测试函数 - 视觉导航任务测试

    Args:
        enable_memory: 是否启用记忆功能进行测试
        frame_dir: 图片帧目录路径
    """
    print("🧪 Embodied Agent Memory System Test")
    print("Testing visual navigation with memory functionality")

    # 设置图片目录
    frame_path = Path(frame_dir)

    # 运行导航任务测试
    result = solve_navigation_with_memory(enable_memory, frame_path)

    if result is None:
        print("❌ Test failed: Could not load images")
        return None

    # 输出结果摘要
    print(f"\n{'='*80}")
    print("VISUAL NAVIGATION TEST RESULTS")
    print(f"{'='*80}")

    print(f"Memory Enabled: {result['memory_enabled']}")
    print(f"Task Success: {result['success']}")
    print(f"Images Used: {result['images_used']}")
    print(".2f")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Output Length: {len(result['output'])} characters")

    if result.get('memory_stats'):
        print("\nMemory Statistics:")
        stats = result['memory_stats']
        print(f"  - Total Memories: {stats.get('total_memories', 0)}")
        print(f"  - Retrieval Count: {stats.get('retrieval_count', 0)}")
        print(f"  - A-MEM Available: {stats.get('amem_available', False)}")

    return result


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
    print("Proxy_API_BASE:" + os.environ.get("Proxy_API_BASE", "Not Set"))
    print("OPENAI_API_KEY:" + os.environ.get("OPENAI_API_KEY", "Not Set"))
    print("DASHSCOPE_API_KEY:" + os.environ.get("DASHSCOPE_API_KEY", "Not Set"))

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ('true', '1', 'yes', 'on'):
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
        # 默认运行对比测试，使用test/vln目录
        run_comparison_test()


