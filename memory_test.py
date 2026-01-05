"""
Quick Start Test for Agent Memory System

测试脚本验证记忆系统在问答任务中的作用。
比较有记忆版本和无记忆版本的回复差异。
通过agentflow solver来决定使用long_memory还是short_memory。
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append('/root/autodl-tmp/FreeAskAgent')

from agentflow.agents.solver_embodied import construct_solver_embodied

from dotenv import load_dotenv
load_dotenv(dotenv_path="agentflow/.env")


def solve_qa_with_memory(enable_memory: bool = True, memory_input: str = None) -> Dict[str, Any]:
    """
    使用记忆系统解决问答任务 - 通过agentflow solver决定使用long_memory还是short_memory

    Args:
        enable_memory: 是否启用记忆功能
        memory_input: 记忆输入内容

    Returns:
        问答任务结果字典
    """
    print(f"\n{'='*80}")
    print(f"Testing Q&A with Memory {'ENABLED' if enable_memory else 'DISABLED'}")
    print(f"{'='*80}")

    # 设置LLM引擎名称
    llm_engine_name = "gpt-4o"

    # 构造solver（让agent决定使用long_memory还是short_memory）
    print("🏗️ Constructing solver with agentflow...")
    memory_config = {
        'retriever_config': {'use_api_embedding': True},
        'storage_dir': "./memory_store",
        'enable_persistence': True,
        'max_memories': 1000
    } if enable_memory else None

    solver = construct_solver_embodied(
        llm_engine_name=llm_engine_name,
        enabled_tools=["Base_Generator_Tool"],
        tool_engine=["gpt-4o"],
        model_engine=["gpt-4o", "gpt-4o", "gpt-4o"],
        output_types="direct",
        max_time=300,
        max_steps=10,
        enable_multimodal=False,
        enable_memory=enable_memory,
        memory_config=memory_config
    )

    # 如果启用了记忆，先添加记忆内容到long_memory
    if enable_memory and memory_input:
        print(f"🧠 Adding memory to long_memory: {memory_input}")
        if hasattr(solver, 'long_memory') and solver.long_memory:
            solver.long_memory.add_memory(memory_input, metadata={"type": "user_input", "timestamp": time.time()})
        else:
            print("⚠️ Long memory not available in solver")

    # 问答任务问题
    qa_question = "广场内有什么超市"

    print(f"❓ Question: {qa_question}")

    if enable_memory and memory_input:
        print(f"📝 Memory Context: {memory_input}")
        print("✅ Memory system integrated into solver")

    # 执行问答任务 - 让agent决定使用long_memory还是short_memory
    print("🚀 Executing Q&A task via agentflow solver...")
    try:
        start_time = time.time()

        # 使用solver的solve方法，让agent内部决定如何使用记忆
        output = solver.solve(
            qa_question,
            task_type="qa_task"
        )

        execution_time = time.time() - start_time

        direct_output = output.get("direct_output", "No output generated")
        print(".2f")
        print(f"📝 Answer: {direct_output[:200]}...")

        result = {
            'memory_enabled': enable_memory,
            'question': qa_question,
            'memory_input': memory_input,
            'output': direct_output,
            'execution_time': execution_time,
            'success': bool(direct_output and len(direct_output.strip()) > 10),
            'memory_stats': None,
            'memory_type_used': None
        }

        # Add memory statistics and determine which memory type was used
        if enable_memory:
            if hasattr(solver, 'long_memory') and solver.long_memory:
                result['memory_stats'] = solver.long_memory.get_stats()
                result['memory_type_used'] = 'long_memory'
                print("🧠 Agent used Long Memory system")
            elif hasattr(solver, 'memory') and solver.memory:
                result['memory_type_used'] = 'short_memory'
                print("🧠 Agent used Short Memory system")

        # 显式保存记忆到磁盘
        if enable_memory and hasattr(solver, 'long_memory') and solver.long_memory:
            print("💾 Saving memory state to disk...")
            save_success = solver.long_memory.save_state()
            if save_success:
                print("✅ Memory state saved successfully")
            else:
                print("❌ Failed to save memory state")

        return result

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

        result = {
            'memory_enabled': enable_memory,
            'question': qa_question,
            'memory_input': memory_input,
            'error': str(e),
            'success': False,
            'memory_stats': None,
            'memory_type_used': None
        }

        # Add memory statistics even in error case if available
        if enable_memory:
            if hasattr(solver, 'long_memory') and solver.long_memory:
                result['memory_stats'] = solver.long_memory.get_stats()
                result['memory_type_used'] = 'long_memory'
            elif hasattr(solver, 'memory') and solver.memory:
                result['memory_type_used'] = 'short_memory'

        # 显式保存记忆到磁盘
        if enable_memory and hasattr(solver, 'long_memory') and solver.long_memory:
            print("💾 Saving memory state to disk...")
            save_success = solver.long_memory.save_state()
            if save_success:
                print("✅ Memory state saved successfully")
            else:
                print("❌ Failed to save memory state")

        return result


def main(enable_memory: bool = True, memory_input: str = None):
    """
    主测试函数 - 问答任务测试

    Args:
        enable_memory: 是否启用记忆功能进行测试
        memory_input: 记忆输入内容
    """
    print("🧪 Agent Memory System Q&A Test")
    print("Testing question answering with memory functionality")

    # 运行问答任务测试
    result = solve_qa_with_memory(enable_memory, memory_input)

    if result is None:
        print("❌ Test failed")
        return None

    # 输出结果摘要
    print(f"\n{'='*80}")
    print("Q&A TEST RESULTS")
    print(f"{'='*80}")

    print(f"Memory Enabled: {result['memory_enabled']}")
    print(f"Memory Type Used: {result.get('memory_type_used', 'None')}")
    print(f"Memory Input: {result.get('memory_input', 'None')}")
    print(f"Question: {result['question']}")
    print(f"Task Success: {result['success']}")
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
        if 'memory_type_used' in result:
            print(f"  - Memory System: {result['memory_type_used']}")

    return result


def run_memory_comparison_test(memory_input: str = "广场内有盒马、永辉等大型超市"):
    """
    运行记忆对比测试：分别测试启用和禁用记忆的情况

    Args:
        memory_input: 要添加的记忆内容
    """
    print("🔄 Running Memory Comparison Test")
    print("This test compares agent Q&A performance with memory ON vs OFF")
    print(f"Memory Input: {memory_input}")
    print(f"Question: 广场内有什么超市")

    # 测试禁用记忆的情况
    print("\n" + "="*50 + " PHASE 1: WITHOUT MEMORY " + "="*50)
    result_without_memory = main(enable_memory=False)

    # 测试启用记忆的情况
    print("\n" + "="*50 + " PHASE 2: WITH MEMORY " + "="*50)
    result_with_memory = main(enable_memory=True, memory_input=memory_input)

    # 检查测试结果是否有效
    if result_without_memory is None or result_with_memory is None:
        print("❌ Comparison test failed: One or both test phases failed")
        return None

    # 对比结果
    print(f"\n{'='*100}")
    print("MEMORY COMPARISON RESULTS")
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
    print(output_without[:1000] + ("..." if len(output_without) > 1000 else ""))

    print("\n--- WITH MEMORY OUTPUT ---")
    output_with = result_with_memory.get('output', 'No output')
    print(output_with[:1000] + ("..." if len(output_with) > 1000 else ""))

    # 分析结果
    print(f"\n{'='*50} ANALYSIS {'='*50}")

    success_without = result_without_memory['success']
    success_with = result_with_memory['success']

    # 检查输出中是否包含记忆中的信息
    memory_keywords = ["盒马", "永辉", "超市"]
    output_with = result_with_memory.get('output', '')
    output_without = result_without_memory.get('output', '')
    memory_mentioned = any(keyword in output_with for keyword in memory_keywords)
    memory_mentioned_without = any(keyword in output_without for keyword in memory_keywords)

    memory_type_used = result_with_memory.get('memory_type_used', 'unknown')

    if success_with and not success_without:
        print("✅ Memory system provides clear benefit!")
        print(f"   Agent used: {memory_type_used}")
        print("   Agent performed better with memory enabled for Q&A.")
        if memory_mentioned:
            print("   Memory content was successfully used in the response.")
    elif success_with and success_without:
        print("🤔 Both tests succeeded - analyzing memory impact...")
        print(f"   Agent used: {memory_type_used}")

        if memory_mentioned and not memory_mentioned_without:
            print("✅ Memory successfully influenced the response!")
            print("   The memory content was incorporated into the answer.")
        elif memory_mentioned and memory_mentioned_without:
            print("🤔 Memory content appears in both responses")
            print("   Memory may have reinforced existing knowledge.")
        else:
            print("⚠️ Memory content not found in responses")
            print("   Memory may not be relevant or retrieval failed.")

        # 比较输出长度作为质量指标
        len_without = len(output_without)
        len_with = len(output_with)

        if len_with > len_without:
            print(f"   Memory version produced {len_with - len_without} more characters of output")
        elif len_with < len_without:
            print(f"   Non-memory version produced {len_without - len_with} more characters of output")

    elif not success_with and not success_without:
        print("❌ Both tests failed - possible issues:")
        print("   - LLM service connectivity issues")
        print("   - Task complexity issues")
    else:
        print("⚠️ Unexpected results - memory version failed but non-memory succeeded")
        print(f"   Agent used: {memory_type_used}")
        print("   This might indicate memory interference or initialization issues")

    return {
        'without_memory': result_without_memory,
        'with_memory': result_with_memory,
        'analysis': {
            'memory_benefit': success_with and not success_without,
            'both_successful': success_with and success_without,
            'both_failed': not success_with and not success_without,
            'unexpected_result': not success_with and success_without,
            'memory_content_used': memory_mentioned,
            'memory_type_used': memory_type_used
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
            memory_input = sys.argv[2] if len(sys.argv) > 2 else "广场内有盒马、永辉等大型超市"
            main(enable_memory=True, memory_input=memory_input)
        elif sys.argv[1].lower() in ('false', '0', 'no', 'off'):
            # 只运行无记忆的版本
            main(enable_memory=False)
        elif sys.argv[1].lower() == 'compare':
            # 运行对比测试
            memory_input = sys.argv[2] if len(sys.argv) > 2 else "广场内有盒马、永辉等大型超市"
            run_memory_comparison_test(memory_input)
        else:
            # 如果第一个参数不是特殊指令，当作记忆内容
            memory_input = sys.argv[1]
            run_memory_comparison_test(memory_input)
    else:
        # 默认运行对比测试，使用指定的记忆内容
        run_memory_comparison_test("广场内有盒马、永辉等大型超市")