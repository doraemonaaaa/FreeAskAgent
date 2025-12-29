# Import the solver
import os
import time
from pathlib import Path

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv(dotenv_path="/root/autodl-tmp/FreeAskAgent/.env")

print("OPENAI_API_BASE:" + os.environ.get("OPENAI_API_BASE", "Not Set"))
print("OPENAI_API_KEY:" + os.environ.get("OPENAI_API_KEY", "Not Set")[:20] + "...")

from agentflow.agentflow.solver_fast import construct_fast_solver
from agentflow.agentflow.solver import construct_solver
from agentflow.models import AgenticMemorySystem  # 导入A-MEM用于预加载记忆

def run_navigation_test(use_memory=False, test_name="Test"):
    """运行导航测试，支持记忆开启/关闭对比"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}: Memory {'ENABLED' if use_memory else 'DISABLED'}")
    print('='*60)

    # Check if we have API keys for real solver
    has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))

    # Always import AgenticMemorySystem for memory functionality
    from agentflow.models import AgenticMemorySystem as A_MEM

    if not has_api_key:
        print("⚠️  未检测到API密钥，使用模拟测试模式")

        # Create mock components for testing memory integration
        from agentflow.models import AgenticMemorySystem

        class MockSolver:
            def __init__(self, use_memory=False):
                self.use_memory = use_memory
                if use_memory:
                    # 配置使用本地模型路径（如果存在的话）
                    local_model_path = "/root/autodl-tmp/all-MiniLM-L6-v2"
                    modules_json_path = "/root/autodl-tmp/all-MiniLM-L6-v2/modules.json"
                    use_local_model = os.path.exists(local_model_path) and os.path.exists(modules_json_path)

                    if use_local_model:
                        print(f"✅ 使用本地模型: {local_model_path}")
                        print(f"✅ 使用本地modules.json: {modules_json_path}")
                        # 强制使用本地模式，禁用所有网络访问
                        os.environ['HF_HUB_OFFLINE'] = '1'  # 让 huggingface 离线模式
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 让 transformers 离线模式
                        os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/root/autodl-tmp'  # 设置本地模型目录
                        retriever_config = {
                            'local_model_path': local_model_path,
                            'modules_json_path': modules_json_path,
                            'use_api_embedding': False,  # 强制使用本地，不使用API
                            'alpha': 0.5,
                            'trust_remote_code': False,  # 不信任远程代码
                            'local_files_only': True  # 只使用本地文件
                        }
                    else:
                        print("⚠️ 本地模型或modules.json不存在，使用API嵌入模式")
                        retriever_config = {
                            'use_api_embedding': True,
                            'alpha': 0.5
                        }
                    self.memory = A_MEM(use_amem=True, retriever_config=retriever_config)
                else:
                    self.memory = A_MEM(use_amem=False)
                self.planner = MockComponent("Planner", use_memory)
                self.verifier = MockComponent("Verifier", use_memory)

        class MockComponent:
            def __init__(self, name, use_memory):
                self.name = name
                self.use_memory = use_memory
                self.historical_memories = []

            def _retrieve_relevant_memories(self, query, k=3):
                if self.use_memory:
                    return ["导航经验：直走遇到路口左转", "问路技巧：选择友善行人", "避障策略：优先后退重规划"]
                return []

            def _get_similar_historical_verifications(self, context, k=2):
                if self.use_memory:
                    return ["导航验证：确认路径无障碍物", "验证历史：类似场景成功"]
                return []

            def add_historical_memory(self, memory):
                self.historical_memories.append(memory)

        solver = MockSolver(use_memory=use_memory)
    else:
        # Use real solver when API keys are available
        FAST_MODE = False

        # Construct the solver
        if FAST_MODE:
            solver = construct_fast_solver(
            llm_engine_name=llm_engine_name,
            enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
            tool_engine=["gpt-4o"],
            base_url=os.environ.get("OPENAI_API_BASE"),
            output_types="direct",
            max_steps=1,
            max_time=10,
            max_tokens=1024,
            fast_max_tokens=256,
            enable_multimodal=True,
            verbose=True
        )
        else:
            solver = construct_solver(
            llm_engine_name=llm_engine_name,
            enabled_tools=["Base_Generator_Tool", "GroundedSAM2_Tool"],
            tool_engine=["gpt-4o"],
            base_url=os.environ.get("OPENAI_API_BASE"),
            model_engine=["gpt-4o", "gpt-4o", "gpt-4o", "gpt-4o"],
            output_types="direct",
            max_time=300,
            max_steps=1,
            enable_multimodal=True,
            use_amem=use_memory,  # 根据参数控制记忆功能
            retriever_config={
                'use_api_embedding': False,  # 测试模式禁用API调用
                'alpha': 0.5
            } if use_memory else None
        )

    # Pre-load navigation memories only if memory is enabled
    if use_memory:
        print("🔍 Pre-loading navigation memories for A-MEM...")
        # 配置使用本地模型路径（如果存在的话）
        local_model_path = "/root/autodl-tmp/all-MiniLM-L6-v2"
        modules_json_path = "/root/autodl-tmp/all-MiniLM-L6-v2/modules.json"
        use_local_model = os.path.exists(local_model_path) and os.path.exists(modules_json_path)

        if use_local_model:
            print(f"✅ 使用本地模型: {local_model_path}")
            print(f"✅ 使用本地modules.json: {modules_json_path}")
            # 强制使用本地模式，禁用所有网络访问
            os.environ['HF_HUB_OFFLINE'] = '1'  # 让 huggingface 离线模式
            os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 让 transformers 离线模式
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/root/autodl-tmp'  # 设置本地模型目录
            retriever_config = {
                'local_model_path': local_model_path,
                'modules_json_path': modules_json_path,
                'use_api_embedding': False,  # 强制使用本地，不使用API
                'alpha': 0.5,
                'trust_remote_code': False,  # 不信任远程代码
                'local_files_only': True  # 只使用本地文件
            }
        else:
            print("⚠️ 本地模型或modules.json不存在，使用API嵌入模式")
            retriever_config = {
                'use_api_embedding': True,
                'alpha': 0.5
            }
        navigation_memory = A_MEM(use_amem=True, retriever_config=retriever_config)

        # Add navigation experience memories
        navigation_memory.add_custom_memory(
            "导航到面包店的经验：从当前位置直走约50米，遇到第一个十字路口左转，然后沿街走200米，面包店通常在右侧有明显招牌的建筑中。",
            memory_type="navigation_experience"
        )

        navigation_memory.add_custom_memory(
            "问路策略：当不确定方向时，可以向路人询问，但要选择看起来友好的行人，使用礼貌语言，注意观察行人的肢体语言判断是否可靠。",
            memory_type="social_interaction"
        )

        navigation_memory.add_custom_memory(
            "避障经验：在城市街道导航时，要特别注意车辆、行人、路边摊位等障碍物。遇到障碍物时优先选择后退重规划，而不是冒险绕行。",
            memory_type="obstacle_avoidance"
        )

        navigation_memory.add_custom_memory(
            "路径记忆：记住成功的导航路径有助于下次更快到达。面包店通常位于商业区，周围有超市、咖啡店等配套设施。",
            memory_type="path_memory"
        )

        print(f"✅ Pre-loaded {navigation_memory.get_stats()['total_memories']} navigation memories")

        # Share memories with solver components
        if hasattr(solver, 'planner') and hasattr(solver.planner, 'add_historical_memory'):
            solver.planner.add_historical_memory("导航经验：直走遇到路口左转能找到面包店")
            solver.planner.add_historical_memory("问路技巧：选择友善的行人询问方向")
            solver.planner.add_historical_memory("避障策略：遇到障碍物优先后退重规划")

        if hasattr(solver, 'verifier') and hasattr(solver.verifier, 'add_verification_memory'):
            solver.verifier.add_verification_memory("导航验证：确认路径中没有明显障碍物，行人指引与地图逻辑一致")

        print("✅ Navigation memories integrated with solver components")

    # Run the navigation task
    try:
        # Check if API keys are available
        has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))

        if not has_api_key:
            print("⚠️  未检测到API密钥，跳过实际LLM调用，仅测试记忆系统集成")
            result = f"测试模式结果 - {test_name} - 记忆系统集成测试完成"
        else:
            try:
                output = solver.solve(
                    navigation_task_prompt,
                    image_paths=image_sequence[:5],  # take up to 5 chronological frames
                )
                result = output.get("direct_output", "No output")
            except Exception as api_error:
                print(f"⚠️  API调用失败 ({str(api_error)[:100]}...)，切换到模拟模式")
                result = f"API错误模拟结果 - {test_name} - API调用失败但记忆系统正常"
        print(f"🎯 Navigation Result ({test_name}):")
        print(result)

        # Analyze memory integration after task completion
        if use_memory:
            print(f"\n🔍 Memory Integration Analysis ({test_name}):")

            # Check planner memory usage
            if hasattr(solver, 'planner'):
                planner_memories = solver.planner._retrieve_relevant_memories("面包店 导航", k=3)
                print(f"📚 Planner retrieved {len(planner_memories)} relevant memories")

                if planner_memories:
                    print("📝 Planner used memories:")
                    for i, mem in enumerate(planner_memories, 1):
                        truncated = mem[:100] + "..." if len(mem) > 100 else mem
                        print(f"   {i}. {truncated}")

            # Check verifier memory usage
            if hasattr(solver, 'verifier'):
                verification_context = "导航到面包店的路径规划验证"
                verifier_memories = solver.verifier._get_similar_historical_verifications(verification_context, k=2)
                print(f"🔍 Verifier retrieved {len(verifier_memories)} historical cases")

            # Check memory accumulation
            if hasattr(solver, 'memory'):
                final_memory_stats = solver.memory.get_stats()
                print(f"🧠 Final memory stats: {final_memory_stats['total_memories']} total memories")

                # Save navigation experience to long-term memory
                if hasattr(solver.memory, 'get_actions'):
                    actions = solver.memory.get_actions()
                    if actions:
                        print(f"💾 Saving {len(actions)} navigation actions to long-term memory")

        return result

    except Exception as e:
        error_msg = f"❌ Error in {test_name}: {str(e)}"
        print(error_msg)
        return error_msg

# Set the LLM engine name (using gpt-4o but with custom API base)
llm_engine_name = "gpt-4o"

# A-MEM Test Mode - set to True to enable full A-MEM testing
AMEM_TEST_MODE = True

# Memory Comparison Test - run both with and without memory，对比测试功能
RUN_MEMORY_COMPARISON = True
#RUN_MEMORY_COMPARISON = False

# Define navigation task prompt
navigation_task_prompt = """"
[Task]
请描述视野中的可行动作并选出后续一连串的导航轨迹指令
你要去面包店
[Rules]
要躲避物体不要撞上
当你离人2m内的时候就可以触发问路
[Policy]
使用最快获取信息的策略，你可选择自己不断探索地点，也可以问人来快速获取信息，尽管可能不精准
[Action Space]
动作空间是[前进，左转，右转，后转, 后退, 停止, 问路][1m, 2m, 3m]
每次动作只能选择一个动作和一个距离, 比如'前进2m'
[Output Format]
请给出后续5步的导航指令序列。
[Tools]
你可以使用GroundedSAM2_Tool来识别图像中的物体，自己设置prompt比如obst，获取物体的位置和类别信息，辅助你做出导航决策。你可以获取obstacle,street,building等信息
[Image Sequence]
这里有一系列按时间顺序排列的图像帧，展示了你当前的视野。请根据这些图像帧来理解环境。
"""

# Prepare an ordered image sequence so the agent can perceive motion
frame_dir = Path("/root/autodl-tmp/FreeAskAgent")
# image_sequence = sorted(str(path) for path in frame_dir.glob("frame_*.jpeg"))
image_sequence = None
if not image_sequence:
    image_sequence = ["/root/autodl-tmp/FreeAskAgent/input_img1.jpg"]

# Main execution logic
if RUN_MEMORY_COMPARISON:
    print("🚀 运行记忆功能对比测试")
    print("="*80)

    # Test 1: Run without memory
    result_without_memory = run_navigation_test(use_memory=False, test_name="无记忆版本")

    print("\n" + "⏸️"*40 + " 等待5秒后开始有记忆版本 " + "⏸️"*40)
    import time
    time.sleep(5)

    # Test 2: Run with memory
    result_with_memory = run_navigation_test(use_memory=True, test_name="有记忆版本")

    # Compare results
    print("\n" + "="*80)
    print("📊 对比分析结果")
    print("="*80)

    print("🔸 无记忆版本结果长度:", len(result_without_memory))
    print("🔸 有记忆版本结果长度:", len(result_with_memory))

    # Check for qualitative differences
    memory_indicators = ["记忆", "经验", "历史", "之前", "上次", "导航经验"]
    memory_mentions_without = sum(1 for indicator in memory_indicators if indicator in result_without_memory.lower())
    memory_mentions_with = sum(1 for indicator in memory_indicators if indicator in result_with_memory.lower())

    print(f"🔸 无记忆版本提及记忆相关词: {memory_mentions_without} 次")
    print(f"🔸 有记忆版本提及记忆相关词: {memory_mentions_with} 次")

    if memory_mentions_with > memory_mentions_without:
        print("✅ 有记忆版本使用了历史经验！")
    else:
        print("⚠️ 记忆使用情况需要进一步分析")

    print("\n🎯 测试完成！请对比两个版本的输出质量和决策逻辑.")
print("\n🎯 测试完成！请对比两个版本的输出质量和决策逻辑.")
