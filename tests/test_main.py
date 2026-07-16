"""
Comprehensive unit tests for agent-framework main.py.

Tests all 10 core functions:
  - react_reasoning_loop
  - chain_of_thought_solver
  - few_shot_template_engine
  - agent_workflow_orchestrator
  - tool_registry_and_dispatcher
  - memory_manager_short_long_term
  - prompt_optimizer
  - multi_agent_coordinator
  - response_parser_and_validator
  - agent_evaluation_metrics
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    react_reasoning_loop,
    chain_of_thought_solver,
    few_shot_template_engine,
    agent_workflow_orchestrator,
    tool_registry_and_dispatcher,
    memory_manager_short_long_term,
    prompt_optimizer,
    multi_agent_coordinator,
    response_parser_and_validator,
    agent_evaluation_metrics,
)


# =====================================================================
# 1. react_reasoning_loop 测试
# =====================================================================

class TestReactReasoningLoop:
    """测试 ReAct 推理循环。"""

    def test_basic_reasoning_with_tool(self):
        """基本推理循环：有工具可用时应调用工具并生成答案。"""
        def search_tool(query):
            return f"Result for: {query}"

        tools = {"search": search_tool}
        result = react_reasoning_loop("What is AI?", tools, max_iterations=5)

        assert "answer" in result
        assert "reasoning_chain" in result
        assert "tool_calls" in result
        assert "iterations" in result
        assert result["iterations"] <= 5
        assert len(result["reasoning_chain"]) > 0

    def test_empty_tools_produces_answer(self):
        """空工具集时应直接生成答案并退出。"""
        tools = {}
        result = react_reasoning_loop("What is 2+2?", tools, max_iterations=5)

        assert "answer" in result
        assert result["answer"] is not None
        # Should exit early with tools empty
        assert result["iterations"] <= 2

    def test_tool_call_limit_safety(self):
        """工具调用次数应有上限保护，防止无限循环。"""
        call_count = [0]

        def infinite_tool(query):
            call_count[0] += 1
            return f"called {call_count[0]}"

        tools = {"loop": infinite_tool}
        result = react_reasoning_loop("test", tools, max_iterations=3)

        # max_tool_calls = max_iterations * 2 = 6; with max_iterations=3,
        # should complete before the safety limit is hit
        assert "answer" in result
        assert result["iterations"] <= 3

    def test_reasoning_chain_structure(self):
        """推理链应包含 thought 和 observation 记录。"""
        def dummy_tool(x):
            return "data"

        tools = {"dummy": dummy_tool}
        result = react_reasoning_loop("Hello", tools, max_iterations=3)

        chain = result["reasoning_chain"]
        types = [entry["type"] for entry in chain]
        assert "thought" in types

    def test_single_iteration_with_sufficient(self):
        """当_generate_thought立即标记为sufficient时，应在1次迭代内完成。"""
        tools = {}
        result = react_reasoning_loop("simple query", tools, max_iterations=10)

        assert result["iterations"] <= 2


# =====================================================================
# 2. chain_of_thought_solver 测试
# =====================================================================

class TestChainOfThoughtSolver:
    """测试链式思维求解器。"""

    def test_simple_chain_passes(self):
        """简单链式步骤应全部通过验证。"""
        def verify_non_empty(x):
            return isinstance(x, str) and len(x) > 0

        steps = [
            {"description": "Step 1", "verify": verify_non_empty},
            {"description": "Step 2", "verify": verify_non_empty},
        ]
        result = chain_of_thought_solver("Problem?", steps)

        assert result["verified"] is True
        assert result["solution"] is not None
        assert len(result["trace"]) >= 2

    def test_no_verify_steps(self):
        """无验证函数的步骤应自动通过。"""
        steps = [
            {"description": "Step A"},
            {"description": "Step B"},
        ]
        result = chain_of_thought_solver("Test", steps)

        assert result["verified"] is True

    def test_backtrack_on_failure(self):
        """验证失败时应回溯到前一步。"""
        def always_fail(x):
            return False

        steps = [
            {"description": "Pass step", "verify": lambda x: True},
            {"description": "Fail step", "verify": always_fail},
        ]
        result = chain_of_thought_solver("Test", steps)

        # Should have backtracked
        assert result["backtracks"] > 0

    def test_first_step_failure_aborts(self):
        """第一步验证失败且无法回溯时应返回失败。"""
        def always_fail(x):
            return False

        steps = [
            {"description": "Fail step", "verify": always_fail},
        ]
        result = chain_of_thought_solver("Test", steps)

        assert result["verified"] is False
        assert "error" in result

    def test_retry_limit(self):
        """重试次数限制应被遵守。"""
        call_counter = [0]

        def verify_count(x):
            call_counter[0] += 1
            return False

        steps = [
            {"description": "Retry step", "verify": verify_count, "retry_limit": 2},
        ]
        result = chain_of_thought_solver("Test", steps)

        # 2 retries + 1 initial = 3 total calls
        # But since it's the first step and all fail, it backtracks but can't
        # Actually with 1 step, first step failure at step 0: cannot backtrack
        assert not result["verified"]


# =====================================================================
# 3. few_shot_template_engine 测试
# =====================================================================

class TestFewShotTemplateEngine:
    """测试 Few-shot 模板引擎。"""

    def test_selects_relevant_examples(self):
        """应选择与查询最相关的示例。"""
        examples = [
            {"input": "The weather is nice today", "output": "positive"},
            {"input": "I am very sad", "output": "negative"},
            {"input": "The weather is great for outdoor activities", "output": "positive"},
            {"input": "I love programming", "output": "positive"},
        ]
        query = "The weather is nice for sports"
        result = few_shot_template_engine("classification", examples, query)

        assert "prompt" in result
        assert "selected_examples" in result
        assert len(result["selected_examples"]) <= 3
        assert len(result["selected_examples"]) > 0

    def test_empty_examples(self):
        """空示例列表应返回基本提示。"""
        result = few_shot_template_engine("qa", [], "What is AI?")

        assert result["selected_examples"] == []
        assert "Query" in result["prompt"]

    def test_single_example(self):
        """单示例时应返回该示例。"""
        examples = [{"input": "test input", "output": "test output"}]
        result = few_shot_template_engine("test", examples, "test input")

        assert len(result["selected_examples"]) == 1

    def test_similarity_scores_included(self):
        """应返回相似度分数列表。"""
        examples = [
            {"input": "apple banana cherry", "output": "A"},
            {"input": "dog cat mouse", "output": "B"},
        ]
        result = few_shot_template_engine("test", examples, "apple banana fruit")

        assert "similarity_scores" in result
        assert len(result["similarity_scores"]) > 0

    def test_prompt_structure(self):
        """生成的提示应包含任务类型和示例。"""
        examples = [
            {"input": "hello world", "output": "greeting"},
        ]
        result = few_shot_template_engine("classification", examples, "hello")

        prompt = result["prompt"]
        assert "Task Type" in prompt
        assert "Examples" in prompt
        assert "Now solve" in prompt


# =====================================================================
# 4. agent_workflow_orchestrator 测试
# =====================================================================

class TestAgentWorkflowOrchestrator:
    """测试 DAG 工作流编排器。"""

    def test_linear_dag(self):
        """线性依赖 DAG 应按正确顺序执行。"""
        def task_a(prev): return "A"
        def task_b(prev): return f"B-{prev.get('a', '')}"

        tasks = {
            "a": {"name": "A", "action": task_a},
            "b": {"name": "B", "action": task_b},
        }
        deps = [("a", "b")]

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "completed"
        assert result["execution_order"] == ["a", "b"]

    def test_parallel_tasks(self):
        """无依赖任务应可并行执行（同批次）。"""
        def make_task(val):
            return lambda p: val

        tasks = {
            "t1": {"name": "T1", "action": make_task("1")},
            "t2": {"name": "T2", "action": make_task("2")},
        }
        deps = []

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "completed"
        assert set(result["execution_order"][:2]) == {"t1", "t2"}

    def test_cycle_detection(self):
        """循环依赖应被检测到并返回错误。"""
        tasks = {
            "a": {"name": "A"},
            "b": {"name": "B"},
        }
        deps = [("a", "b"), ("b", "a")]

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "error"
        assert "循环依赖" in result.get("error", "")

    def test_condition_branch_skip(self):
        """条件分支为 False 时应跳过任务及其后继。"""

        def task_a(prev): return {"value": 1}
        def cond_b(results):
            return results.get("a", {}).get("value", 0) > 10  # False

        tasks = {
            "a": {"name": "A", "action": task_a},
            "b": {"name": "B", "action": lambda p: "B", "condition": cond_b},
            "c": {"name": "C", "action": lambda p: "C"},
        }
        deps = [("a", "b"), ("b", "c")]

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "completed"
        assert "b" in result["skipped"]
        assert "c" in result["skipped"]

    def test_missing_task_id_in_dependencies(self):
        """依赖中包含不存在的任务ID时应返回错误而非 KeyError。"""
        tasks = {
            "a": {"name": "A"},
        }
        deps = [("a", "nonexistent")]

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "error"
        assert "未定义的任务ID" in result.get("error", "")

    def test_task_action_error_handling(self):
        """任务执行出错时不应中断整个流程。"""
        def failing_action(prev):
            raise ValueError("test error")

        def ok_action(prev):
            return "OK"

        tasks = {
            "a": {"name": "A", "action": failing_action},
            "b": {"name": "B", "action": ok_action},
        }
        deps = [("a", "b")]

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "completed"
        assert "error" in result["results"]["a"]

    def test_no_deps_no_agents(self):
        """无依赖、无Agent的最简场景应正常执行。"""
        tasks = {
            "t": {"name": "T"},
        }
        deps = []

        result = agent_workflow_orchestrator(tasks, deps, {})
        assert result["status"] == "completed"
        assert result["execution_order"] == ["t"]


# =====================================================================
# 5. tool_registry_and_dispatcher 测试
# =====================================================================

class TestToolRegistryAndDispatcher:
    """测试工具注册与分派系统。"""

    def test_basic_dispatch(self):
        """基本工具调用应返回正确结果。"""
        registry = {
            "add": {
                "function": lambda x, y: x + y,
                "parameters": {
                    "x": {"type": "int", "required": True},
                    "y": {"type": "int", "required": True},
                },
                "cache": False,
            }
        }
        result = tool_registry_and_dispatcher("add", {"x": 10, "y": 20}, registry)
        assert result["result"] == 30
        assert result["cached"] is False
        assert result["error"] is None

    def test_unregistered_tool(self):
        """未注册的工具应返回错误。"""
        registry = {}
        result = tool_registry_and_dispatcher("unknown", {}, registry)
        assert result["result"] is None
        assert "未注册" in result["error"]

    def test_missing_required_param(self):
        """缺少必需参数应返回错误。"""
        registry = {
            "greet": {
                "function": lambda name: f"Hello {name}",
                "parameters": {"name": {"type": "str", "required": True}},
            }
        }
        result = tool_registry_and_dispatcher("greet", {}, registry)
        assert result["result"] is None
        assert "缺少必需参数" in result["error"]

    def test_type_validation(self):
        """参数类型校验应捕获类型不匹配。"""
        registry = {
            "double": {
                "function": lambda x: x * 2,
                "parameters": {"x": {"type": "int", "required": True}},
            }
        }
        result = tool_registry_and_dispatcher("double", {"x": "not_int"}, registry)
        assert result["result"] is None
        assert "int 类型" in result["error"]

    def test_cache_hit(self):
        """缓存命中时应返回缓存结果而不重复执行。"""
        call_record = []

        def tracked_func(x, y):
            call_record.append(1)
            return x * y

        registry = {
            "multiply": {
                "function": tracked_func,
                "parameters": {
                    "x": {"type": "int", "required": True},
                    "y": {"type": "int", "required": True},
                },
                "cache": True,
            }
        }
        # First call
        r1 = tool_registry_and_dispatcher("multiply", {"x": 3, "y": 4}, registry)
        assert r1["result"] == 12
        assert r1["cached"] is False
        assert len(call_record) == 1

        # Second call with same args
        r2 = tool_registry_and_dispatcher("multiply", {"x": 3, "y": 4}, registry)
        assert r2["result"] == 12
        assert r2["cached"] is True
        assert len(call_record) == 1  # Not called again

    def test_non_dict_args_protection(self):
        """非 dict 类型的 args 应返回错误而非崩溃。"""
        registry = {
            "test": {
                "function": lambda x: x,
                "parameters": {},
            }
        }
        result = tool_registry_and_dispatcher("test", "not_a_dict", registry)
        assert result["result"] is None
        assert "dict" in result["error"].lower() or "dict" in result["error"]

    def test_func_execution_error(self):
        """工具执行异常应被捕获并返回错误信息。"""
        def error_func(x):
            raise RuntimeError("test error message")

        registry = {
            "fail": {
                "function": error_func,
                "parameters": {"x": {"type": "int", "required": True}},
            }
        }
        result = tool_registry_and_dispatcher("fail", {"x": 1}, registry)
        assert result["result"] is None
        assert "test error message" in result["error"]


# =====================================================================
# 6. memory_manager_short_long_term 测试
# =====================================================================

class TestMemoryManager:
    """测试记忆管理系统。"""

    def test_add_and_retrieve_short(self):
        """添加并检索短期记忆。"""
        r = memory_manager_short_long_term("add_short", "Hello world", {})
        assert r["short_term_size"] >= 1

        r = memory_manager_short_long_term("retrieve_short", "", {})
        assert r["count"] >= 1
        assert "Hello world" in r["memories"]

    def test_short_memory_capacity(self):
        """短期记忆应遵守容量限制，淘汰旧条目。"""
        for i in range(110):
            memory_manager_short_long_term("add_short", f"item_{i}", {"capacity": 100})

        stats = memory_manager_short_long_term("stats", "", {})
        assert stats["short_term_count"] <= 100

    def test_add_and_retrieve_long(self):
        """添加并检索长期记忆（TF-IDF）。"""
        memory_manager_short_long_term("add_long", "Python is a programming language", {})
        memory_manager_short_long_term("add_long", "Machine learning is a subset of AI", {})
        memory_manager_short_long_term("add_long", "Java is also a language", {})

        result = memory_manager_short_long_term(
            "retrieve_long", "programming language Python", {"top_k": 3}
        )
        assert "results" in result
        assert len(result["results"]) > 0

    def test_tfidf_zero_query_norm(self):
        """当查询词无法分词时应安全处理（不除零）。"""
        # Add some docs first
        memory_manager_short_long_term("add_long", "test document", {})
        # Empty query should produce no tokens
        result = memory_manager_short_long_term("retrieve_long", "   ", {"top_k": 1})

        assert "results" in result

    def test_forget_short_memory(self):
        """遗忘短期记忆条目。"""
        r = memory_manager_short_long_term("add_short", "temp", {})
        key = r["key"]

        r = memory_manager_short_long_term("forget", key, {})
        assert r["forgotten"] == key

    def test_forget_nonexistent(self):
        """遗忘不存在的 key 应返回消息。"""
        r = memory_manager_short_long_term("forget", "nonexistent_key", {})
        assert r["forgotten"] is None

    def test_stats(self):
        """统计信息应包含各项计数。"""
        stats = memory_manager_short_long_term("stats", "", {})
        assert "short_term_count" in stats
        assert "long_term_count" in stats
        assert "vocabulary_size" in stats
        assert "total_docs" in stats

    def test_unknown_operation(self):
        """未知操作应返回错误。"""
        r = memory_manager_short_long_term("invalid_op", "", {})
        assert "error" in r


# =====================================================================
# 7. prompt_optimizer 测试
# =====================================================================

class TestPromptOptimizer:
    """测试提示词优化器。"""

    def test_clarity_check(self):
        """模糊性检查应检测模糊词汇。"""
        result = prompt_optimizer("帮我大概写一下文章", ["clarity"])
        assert result["improvements_count"] >= 1
        assert len(result["suggestions"]) >= 1

    def test_role_check_adds_role(self):
        """缺少角色定义时应添加默认角色。"""
        result = prompt_optimizer("write an article", ["role"])
        optimized = result["optimized_prompt"]
        assert "AI助手" in optimized or "assistant" in optimized.lower()
        # The suggestion is in Chinese: "缺少角色定义"
        assert len(result["suggestions"]) >= 1

    def test_format_check(self):
        """缺少格式要求时应添加格式提示。"""
        result = prompt_optimizer("analyze this", ["format"])
        assert "结构化格式" in result["optimized_prompt"] or "format" in result["optimized_prompt"].lower()

    def test_specificity_short_prompt(self):
        """过短的提示词应被标记。"""
        result = prompt_optimizer("hi", ["specificity"])
        assert result["score"] < 80

    def test_context_check(self):
        """缺少上下文时应提示。"""
        result = prompt_optimizer("do it", ["context"])
        assert len(result["suggestions"]) >= 1

    def test_all_goals(self):
        """所有优化目标应同时工作。"""
        goals = ["clarity", "context", "role", "format", "specificity"]
        result = prompt_optimizer("write something", goals)
        assert "optimized_prompt" in result
        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_empty_prompt(self):
        """空提示词应返回有效结果。"""
        result = prompt_optimizer("", ["clarity", "format"])
        assert "optimized_prompt" in result
        assert isinstance(result["score"], (int, float))


# =====================================================================
# 8. multi_agent_coordinator 测试
# =====================================================================

class TestMultiAgentCoordinator:
    """测试多 Agent 协调器 — 三种协议。"""

    def setup_agents(self):
        return [
            {"id": "a1", "name": "Alpha", "role": "manager", "capability": 0.9},
            {"id": "a2", "name": "Beta", "role": "worker", "capability": 0.7},
            {"id": "a3", "name": "Gamma", "role": "worker", "capability": 0.6},
        ]

    def test_cooperative_protocol(self):
        """协作模式应按能力分配子任务。"""
        agents = self.setup_agents()
        task = {"description": "Build report", "subtasks": ["Research", "Draft", "Review"]}

        result = multi_agent_coordinator(agents, task, "cooperative")
        assert result["protocol"] == "cooperative"
        assert len(result["assignments"]) == 3
        assert len(result["messages"]) >= 3

    def test_competitive_protocol(self):
        """竞争模式应让所有 Agent 竞争并选出胜者。"""
        agents = self.setup_agents()
        task = {"description": "Write best answer"}

        result = multi_agent_coordinator(agents, task, "competitive")
        assert result["protocol"] == "competitive"
        assert len(result["assignments"]) == len(agents)
        assert any(m["type"] == "result_announcement" for m in result["messages"])

    def test_hierarchical_protocol(self):
        """层级模式应按角色分级分配任务。"""
        agents = self.setup_agents()
        task = {"description": "Project plan", "subtasks": ["Plan", "Execute"]}

        result = multi_agent_coordinator(agents, task, "hierarchical")
        assert result["protocol"] == "hierarchical"
        assert len(result["assignments"]) >= 1
        directive_msgs = [m for m in result["messages"] if m["type"] == "directive"]
        assert len(directive_msgs) >= 1

    def test_unknown_protocol(self):
        """未知协议应返回错误。"""
        agents = self.setup_agents()
        task = {"description": "test"}

        result = multi_agent_coordinator(agents, task, "unknown_protocol")
        assert "error" in result

    def test_single_agent(self):
        """单 Agent 场景应正常处理。"""
        agents = [{"id": "solo", "name": "Solo", "role": "worker", "capability": 0.5}]
        task = {"description": "Simple task"}

        result = multi_agent_coordinator(agents, task, "cooperative")
        assert result["agent_count"] == 1
        assert len(result["assignments"]) >= 1

    def test_no_subtasks_cooperative(self):
        """无子任务时协作模式应分配主任务。"""
        agents = self.setup_agents()
        task = {"description": "Main task only"}

        result = multi_agent_coordinator(agents, task, "cooperative")
        assert len(result["assignments"]) >= 1

    def test_message_structure(self):
        """消息应包含正确的 from/to/type/content 结构。"""
        agents = self.setup_agents()
        task = {"description": "test"}

        result = multi_agent_coordinator(agents, task, "cooperative")
        for msg in result["messages"]:
            assert "from" in msg
            assert "to" in msg
            assert "type" in msg
            assert "content" in msg


# =====================================================================
# 9. response_parser_and_validator 测试
# =====================================================================

class TestResponseParserAndValidator:
    """测试响应解析与验证器。"""

    def test_valid_json(self):
        """有效 JSON 应被直接解析。"""
        response = '{"name": "Alice", "age": 30}'
        result = response_parser_and_validator(response, "json")

        assert result["valid"] is True
        assert result["parsed"] == {"name": "Alice", "age": 30}
        assert result["repaired"] is False

    def test_json_in_code_block(self):
        """JSON 包裹在代码块中应被提取并解析。"""
        response = '```json\n{"key": "value"}\n```'
        result = response_parser_and_validator(response, "json")

        assert result["valid"] is True
        assert result["parsed"] == {"key": "value"}
        assert result["repaired"] is True

    def test_json_with_extra_text(self):
        """JSON 前后有额外文本时应提取 JSON 部分。"""
        response = 'Here is the result: {"status": "ok", "count": 42} Thanks!'
        result = response_parser_and_validator(response, "json")

        assert result["valid"] is True
        assert result["repaired"] is True
        assert result["parsed"]["status"] == "ok"

    def test_trailing_comma_repair(self):
        """JSON 尾逗号应被修复。"""
        response = '{"name": "test", "value": 42,}'
        result = response_parser_and_validator(response, "json")

        assert result["valid"] is True
        assert result["repaired"] is True
        assert result["parsed"]["name"] == "test"

    def test_unrepairable_json(self):
        """无法修复的 JSON 应返回无效。"""
        result = response_parser_and_validator("this is not json at all", "json")

        assert result["valid"] is False

    def test_markdown_parsing(self):
        """Markdown 应被解析出标题和链接。"""
        response = """# Title\n## Section\n[link](http://example.com)\n```py\nprint('hi')\n```"""
        result = response_parser_and_validator(response, "markdown")

        assert result["valid"] is True
        assert "headings" in result["parsed"]
        assert len(result["parsed"]["headings"]) == 2
        assert len(result["parsed"]["links"]) == 1

    def test_markdown_no_headings(self):
        """无标题的 Markdown 应有警告。"""
        result = response_parser_and_validator("plain text only", "markdown")
        assert len(result["errors"]) >= 1

    def test_structured_parsing(self):
        """结构化文本应解析 key: value 对。"""
        response = "name: Alice\nage: 30\ncity: Beijing"
        result = response_parser_and_validator(response, "structured")

        assert result["valid"] is True
        assert result["parsed"]["name"] == "Alice"
        assert result["parsed"]["age"] == "30"

    def test_text_format(self):
        """纯文本格式应返回原文本。"""
        response = "Simple text response"
        result = response_parser_and_validator(response, "text")

        assert result["valid"] is True
        assert result["parsed"] == response

    def test_empty_response(self):
        """空响应应有错误。"""
        result = response_parser_and_validator("", "text")
        assert len(result["errors"]) >= 1


# =====================================================================
# 10. agent_evaluation_metrics 测试
# =====================================================================

class TestAgentEvaluationMetrics:
    """测试 Agent 评估指标计算器。"""

    def test_perfect_accuracy(self):
        """完全匹配时应得到 100% 准确率。"""
        outputs = [
            {"answer": "Paris is the capital of France", "response_time": 1.0},
            {"answer": "北京", "response_time": 0.5},
        ]
        truth = [
            {"answer": "Paris is the capital of France"},
            {"answer": "北京"},
        ]
        metrics = agent_evaluation_metrics(outputs, truth)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] > 0.9

    def test_partial_match(self):
        """部分匹配应有中等分数。"""
        outputs = [
            {"answer": "Paris is the capital of France", "response_time": 1.0},
            {"answer": "wrong answer", "response_time": 0.5},
        ]
        truth = [
            {"answer": "Paris is the capital of France"},
            {"answer": "correct answer"},
        ]
        metrics = agent_evaluation_metrics(outputs, truth)

        # First answer matches exactly (1/2 = 0.5)
        assert 0.0 < metrics["accuracy"] < 1.0

    def test_empty_data(self):
        """空数据应返回错误。"""
        metrics = agent_evaluation_metrics([], [])
        assert "error" in metrics

    def test_hallucination_detection(self):
        """幻觉率应正确检测未来源信息。"""
        outputs = [
            {"answer": "Answer 1", "cited_sources": ["src1", "src2"]},
            {"answer": "Answer 2", "cited_sources": ["src3"]},
        ]
        truth = [
            {"answer": "Answer 1", "sources": ["src1"]},
            {"answer": "Answer 2", "sources": ["src3", "src4"]},
        ]
        metrics = agent_evaluation_metrics(outputs, truth)

        # First output cites src2 not in truth -> hallucination
        assert "hallucination_rate" in metrics

    def test_response_time_average(self):
        """响应时间平均值应正确计算。"""
        outputs = [
            {"answer": "A", "response_time": 2.0},
            {"answer": "B", "response_time": 1.0},
            {"answer": "C", "response_time": 3.0},
        ]
        truth = [
            {"answer": "A"}, {"answer": "B"}, {"answer": "C"},
        ]
        metrics = agent_evaluation_metrics(outputs, truth)
        assert metrics["avg_response_time_sec"] == 2.0

    def test_precision_recall_f1_range(self):
        """精确率、召回率、F1 应在 0-1 范围。"""
        outputs = [
            {"answer": "cat dog bird", "response_time": 1.0},
        ]
        truth = [
            {"answer": "cat dog fish"},
        ]
        metrics = agent_evaluation_metrics(outputs, truth)

        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_all_metrics_present(self):
        """所有指标字段应存在。"""
        outputs = [{"answer": "test", "response_time": 1.0}]
        truth = [{"answer": "test"}]
        metrics = agent_evaluation_metrics(outputs, truth)

        expected_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "hallucination_rate", "avg_response_time_sec",
            "total_samples", "correct_count",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
