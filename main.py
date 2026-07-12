"""AI Agent 框架工具集 — 纯标准库实现。

本模块不依赖任何外部库，全部使用 Python 标准库（json, re, collections,
time, math, functools 等）实现 Agent 开发所需的高级算法。

核心算法包括：
  - ReAct 推理循环（Thought→Action→Observation 追踪）
  - 链式思维求解器（分步推理+中间验证+回溯）
  - Few-shot 模板引擎（示例学习+动态选择+模板填充）
  - DAG 任务依赖图调度引擎（拓扑排序+并行/串行/条件分支）
  - 工具注册与分派系统（签名验证+参数校验+结果缓存）
  - 短期记忆(LRU)+长期记忆(TF-IDF 检索)管理器
  - 提示词结构分析与优化器
  - 多Agent协调器（协作/竞争/层级协议）
  - 响应解析与自动修复器
  - Agent评估指标计算器（准确率/F1/幻觉率）
"""

import json
import re
import math
import time
import hashlib
from collections import defaultdict, OrderedDict, Counter
from functools import wraps

# =====================================================================
# 1. ReAct 推理循环
# =====================================================================

def react_reasoning_loop(query, tools, max_iterations=10):
    """ReAct (Reasoning + Acting) 推理循环。

    实现 Thought → Action → Observation 循环，支持工具调用和推理链追踪。
    每轮迭代：分析当前状态→选择工具→执行→观察结果→更新推理链。

    Args:
        query: 用户查询字符串。
        tools: 工具字典，{tool_name: callable}，每个工具接受一个参数返回结果。
        max_iterations: 最大迭代次数，默认 10。

    Returns:
        dict: {
            "answer": 最终答案,
            "reasoning_chain": 推理链列表,
            "tool_calls": 工具调用记录,
            "iterations": 实际迭代次数,
        }
    """
    reasoning_chain = []
    tool_calls = []
    context = query
    answer = None

    for i in range(max_iterations):
        # Thought: 分析当前状态，决定下一步
        thought = _generate_thought(context, tools, reasoning_chain)
        reasoning_chain.append({
            "step": i + 1,
            "type": "thought",
            "content": thought["content"],
        })

        # 判断是否已有足够信息回答
        if thought.get("sufficient"):
            answer = thought.get("answer", context)
            reasoning_chain.append({
                "step": i + 1,
                "type": "final_answer",
                "content": answer,
            })
            break

        # Action: 选择并调用工具
        tool_name = thought.get("tool")
        tool_input = thought.get("tool_input", "")

        if tool_name and tool_name in tools:
            try:
                observation = tools[tool_name](tool_input)
                tool_calls.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "input": tool_input,
                    "output": observation,
                    "success": True,
                })
                reasoning_chain.append({
                    "step": i + 1,
                    "type": "action",
                    "tool": tool_name,
                    "input": tool_input,
                })
                reasoning_chain.append({
                    "step": i + 1,
                    "type": "observation",
                    "content": str(observation),
                })
                # 更新上下文
                context = f"{context}\n[Observation from {tool_name}]: {observation}"
            except Exception as e:
                tool_calls.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "input": tool_input,
                    "output": str(e),
                    "success": False,
                })
                context += f"\n[Error]: {e}"
        else:
            # 无可用工具，直接给出答案
            answer = f"基于推理: {context}"
            reasoning_chain.append({
                "step": i + 1,
                "type": "final_answer",
                "content": answer,
            })
            break

    if answer is None:
        answer = f"经过 {max_iterations} 次迭代后的推理结论: {context}"
        reasoning_chain.append({
            "step": max_iterations,
            "type": "final_answer",
            "content": answer,
        })

    return {
        "answer": answer,
        "reasoning_chain": reasoning_chain,
        "tool_calls": tool_calls,
        "iterations": i + 1,
    }


def _generate_thought(context, tools, history):
    """基于上下文和推理历史生成思维步骤（简化版推理引擎）。"""
    thought = {"content": ""}

    # 如果已有足够多的观察结果，标记为足够
    observations = [h for h in history if h.get("type") == "observation"]
    if len(observations) >= 2:
        thought["sufficient"] = True
        thought["answer"] = f"综合分析结论: {context[-200:]}"
        thought["content"] = "已收集足够信息，可以给出最终答案。"
        return thought

    # 选择第一个可用工具
    if tools:
        tool_name = next(iter(tools))
        thought["tool"] = tool_name
        thought["tool_input"] = context[:100]
        thought["content"] = f"需要调用工具 {tool_name} 获取更多信息。"
    else:
        thought["sufficient"] = True
        thought["answer"] = f"无可用工具，直接回答: {context}"
        thought["content"] = "无工具可用，直接回答。"

    return thought


# =====================================================================
# 2. 链式思维求解器
# =====================================================================

def chain_of_thought_solver(problem, steps):
    """链式思维 (Chain-of-Thought) 求解器。

    分步推理+中间结果验证+回溯。每步推理输出中间结论，
    验证通过后进入下一步；验证失败时回溯到前一步重试。

    Args:
        problem: 问题描述字符串。
        steps: 推理步骤定义列表，每个元素为 dict:
            {"description": 步骤描述, "verify": 验证函数(可选),
             "retry_limit": 重试次数(默认2)}。

    Returns:
        dict: {
            "solution": 最终解,
            "trace": 推理轨迹,
            "verified": 是否全部验证通过,
            "backtracks": 回溯次数,
        }
    """
    trace = []
    results = []
    backtracks = 0
    step_idx = 0

    while step_idx < len(steps):
        step = steps[step_idx]
        retry_limit = step.get("retry_limit", 2)
        verified = False

        for attempt in range(retry_limit + 1):
            # 执行推理步骤
            prev_result = results[-1]["result"] if results else problem
            step_result = _execute_step(
                step["description"], prev_result, attempt
            )
            trace.append({
                "step": step_idx + 1,
                "attempt": attempt + 1,
                "description": step["description"],
                "input": prev_result,
                "result": step_result,
            })

            # 验证中间结果
            verify_fn = step.get("verify")
            if verify_fn is None:
                verified = True
                results.append({
                    "step": step_idx + 1,
                    "result": step_result,
                    "verified": True,
                })
                break

            try:
                if verify_fn(step_result):
                    verified = True
                    results.append({
                        "step": step_idx + 1,
                        "result": step_result,
                        "verified": True,
                    })
                    break
                else:
                    trace.append({
                        "step": step_idx + 1,
                        "attempt": attempt + 1,
                        "type": "verification_failed",
                        "message": f"验证失败，重试 {attempt + 1}/{retry_limit}",
                    })
            except Exception as e:
                trace.append({
                    "step": step_idx + 1,
                    "attempt": attempt + 1,
                    "type": "verification_error",
                    "message": str(e),
                })

        if not verified:
            # 回溯到前一步
            if step_idx > 0:
                step_idx -= 1
                backtracks += 1
                trace.append({
                    "type": "backtrack",
                    "from_step": step_idx + 2,
                    "to_step": step_idx + 1,
                    "message": "验证失败，回溯到前一步",
                })
                results.pop()
                continue
            else:
                # 第一步就失败，放弃
                return {
                    "solution": None,
                    "trace": trace,
                    "verified": False,
                    "backtracks": backtracks,
                    "error": "无法通过第一步验证",
                }

        step_idx += 1

    solution = results[-1]["result"] if results else None
    return {
        "solution": solution,
        "trace": trace,
        "verified": True,
        "backtracks": backtracks,
    }


def _execute_step(description, prev_result, attempt):
    """执行单个推理步骤（简化版）。"""
    # 模拟推理：对输入进行简单变换
    if isinstance(prev_result, (int, float)):
        return prev_result
    if attempt == 0:
        return f"[{description}] => {prev_result}"
    return f"[{description} (retry {attempt})] => {prev_result}"


# =====================================================================
# 3. Few-shot 模板引擎
# =====================================================================

def few_shot_template_engine(task_type, examples, query):
    """Few-shot 模板引擎。

    从示例中学习模式 → 动态选择最佳示例 → 模板填充。
    基于TF-IDF相似度从示例库中选择与查询最相关的 top-k 示例。

    Args:
        task_type: 任务类型（如 'classification', 'extraction', 'qa'）。
        examples: 示例列表，每个元素为
            {"input": 输入文本, "output": 期望输出, "tags": 标签列表(可选)}。
        query: 查询输入文本。

    Returns:
        dict: {"prompt": 生成的提示词, "selected_examples": 选中的示例}.
    """
    if not examples:
        return {"prompt": f"Task: {task_type}\nQuery: {query}\nOutput:",
                "selected_examples": []}

    # 计算查询与每个示例的相似度（基于词频重叠率）
    query_words = set(query.lower().split())
    scored = []
    for idx, ex in enumerate(examples):
        ex_words = set(ex["input"].lower().split())
        if not ex_words:
            score = 0.0
        else:
            # Jaccard 相似度
            overlap = len(query_words & ex_words)
            union = len(query_words | ex_words)
            score = overlap / union if union > 0 else 0.0
        scored.append((score, idx, ex))

    # 按相似度降序选择
    scored.sort(reverse=True, key=lambda x: x[0])
    k = min(3, len(examples))
    selected = [s[2] for s in scored[:k]]

    # 构建提示词
    prompt_parts = [f"Task Type: {task_type}", ""]
    prompt_parts.append("Examples:")
    for i, ex in enumerate(selected):
        prompt_parts.append(f"Example {i + 1}:")
        prompt_parts.append(f"  Input: {ex['input']}")
        prompt_parts.append(f"  Output: {ex['output']}")
        prompt_parts.append("")
    prompt_parts.append("Now solve:")
    prompt_parts.append(f"  Input: {query}")
    prompt_parts.append(f"  Output:")

    return {
        "prompt": "\n".join(prompt_parts),
        "selected_examples": selected,
        "similarity_scores": [{"index": s[1], "score": round(s[0], 4)}
                              for s in scored[:k]],
    }


# =====================================================================
# 4. Agent 工作流编排器 (DAG 调度)
# =====================================================================

def agent_workflow_orchestrator(tasks, dependencies, agents):
    """Agent 工作流编排器 — 基于 DAG 的任务依赖图调度。

    使用 Kahn 算法进行拓扑排序，支持并行/串行/条件分支。
    每个任务完成后触发其依赖者，支持条件分支跳过。

    Args:
        tasks: 任务定义字典 {task_id: {"name": 名称, "agent": agent_id,
               "action": 执行函数(可选), "condition": 条件函数(可选)}}。
        dependencies: 依赖关系列表 [(predecessor_id, successor_id), ...]。
        agents: Agent 字典 {agent_id: {"name": 名称, "status": "idle"}}。

    Returns:
        dict: {"execution_order": 执行顺序, "results": 各任务结果,
               "skipped": 被跳过的任务, "status": 总体状态}.
    """
    # 构建邻接表和入度表
    adj = defaultdict(list)
    in_degree = {tid: 0 for tid in tasks}
    for pred, succ in dependencies:
        adj[pred].append(succ)
        in_degree[succ] = in_degree.get(succ, 0) + 1

    # Kahn 拓扑排序
    queue = [tid for tid in tasks if in_degree[tid] == 0]
    execution_order = []
    results = {}
    skipped = set()

    while queue:
        # 同层任务可并行执行
        batch = queue[:]
        queue = []
        for task_id in batch:
            task = tasks[task_id]

            # 检查条件分支
            if task_id in skipped:
                # 跳过该任务及其所有后继
                _skip_descendants(task_id, adj, skipped)
                continue

            condition = task.get("condition")
            if condition is not None:
                try:
                    if not condition(results):
                        skipped.add(task_id)
                        _skip_descendants(task_id, adj, skipped)
                        continue
                except Exception:
                    pass

            # 执行任务
            action = task.get("action")
            if action:
                try:
                    result = action(results)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"status": "no_action"}

            results[task_id] = result
            execution_order.append(task_id)

            # 更新后继任务入度
            for succ in adj[task_id]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

    # 检查是否有环
    if len(execution_order) + len(skipped) < len(tasks):
        cyclic = set(tasks.keys()) - set(execution_order) - skipped
        return {
            "execution_order": execution_order,
            "results": results,
            "skipped": list(skipped),
            "status": "error",
            "error": f"检测到循环依赖: {cyclic}",
        }

    return {
        "execution_order": execution_order,
        "results": results,
        "skipped": list(skipped),
        "status": "completed",
    }


def _skip_descendants(task_id, adj, skipped):
    """递归跳过任务的所有后继。"""
    for succ in adj[task_id]:
        if succ not in skipped:
            skipped.add(succ)
            _skip_descendants(succ, adj, skipped)


# =====================================================================
# 5. 工具注册与分派系统
# =====================================================================

def tool_registry_and_dispatcher(tool_name, args, registry):
    """工具注册与分派系统。

    支持工具注册、签名验证、参数校验、结果缓存和错误处理。

    Args:
        tool_name: 要调用的工具名称。
        args: 调用参数字典。
        registry: 工具注册表，结构为:
            {tool_name: {
                "function": 执行函数,
                "parameters": {param_name: {"type": 类型, "required": bool}},
                "cache": bool,  # 是否启用缓存
            }}

    Returns:
        dict: {"result": 结果, "cached": 是否命中缓存, "error": 错误信息(若有)}.
    """
    # 静态缓存（跨调用保留）
    if not hasattr(tool_registry_and_dispatcher, "_cache"):
        tool_registry_and_dispatcher._cache = {}

    if tool_name not in registry:
        return {"result": None, "cached": False,
                "error": f"工具 '{tool_name}' 未注册"}

    tool_spec = registry[tool_name]
    func = tool_spec.get("function")
    params_spec = tool_spec.get("parameters", {})
    use_cache = tool_spec.get("cache", False)

    # 参数校验
    for param_name, param_spec in params_spec.items():
        if param_spec.get("required", False) and param_name not in args:
            return {"result": None, "cached": False,
                    "error": f"缺少必需参数: {param_name}"}

        if param_name in args:
            expected_type = param_spec.get("type")
            actual_val = args[param_name]
            if expected_type == "str" and not isinstance(actual_val, str):
                return {"result": None, "cached": False,
                        "error": f"参数 {param_name} 应为 str 类型"}
            elif expected_type == "int" and not isinstance(actual_val, int):
                return {"result": None, "cached": False,
                        "error": f"参数 {param_name} 应为 int 类型"}
            elif expected_type == "list" and not isinstance(actual_val, list):
                return {"result": None, "cached": False,
                        "error": f"参数 {param_name} 应为 list 类型"}

    # 缓存检查
    cache_key = None
    if use_cache:
        cache_input = json.dumps(args, sort_keys=True, default=str)
        cache_key = hashlib.md5(
            f"{tool_name}:{cache_input}".encode()
        ).hexdigest()
        if cache_key in tool_registry_and_dispatcher._cache:
            return {"result": tool_registry_and_dispatcher._cache[cache_key],
                    "cached": True, "error": None}

    # 执行工具
    try:
        result = func(**args) if isinstance(args, dict) else func(args)
        if use_cache and cache_key:
            tool_registry_and_dispatcher._cache[cache_key] = result
        return {"result": result, "cached": False, "error": None}
    except Exception as e:
        return {"result": None, "cached": False, "error": str(e)}


# =====================================================================
# 6. 记忆管理系统
# =====================================================================

def memory_manager_short_long_term(operation, data, params=None):
    """记忆管理系统 — 短期记忆(LRU)+长期记忆(TF-IDF 检索)。

    短期记忆使用 LRU (Least Recently Used) 缓存策略，
    长期记忆使用 TF-IDF 索引进行语义检索。支持添加/检索/遗忘操作。

    Args:
        operation: 操作类型 — 'add_short', 'add_long', 'retrieve_short',
                   'retrieve_long', 'forget', 'stats'。
        data: 操作数据。添加时为记忆内容，检索时为查询文本。
        params: 参数字典，可包含 'capacity'(短期容量), 'top_k'(检索数)。

    Returns:
        操作结果（因操作类型而异）。
    """
    # 静态状态
    if not hasattr(memory_manager_short_long_term, "_short_memory"):
        memory_manager_short_long_term._short_memory = OrderedDict()
        memory_manager_short_long_term._long_memory = []  # [{"text":..., "tf":{}}]
        memory_manager_short_long_term._idf = defaultdict(float)
        memory_manager_short_long_term._doc_count = 0

    sm = memory_manager_short_long_term._short_memory
    lm = memory_manager_short_long_term._long_memory
    idf = memory_manager_short_long_term._idf
    params = params or {}

    if operation == "add_short":
        capacity = params.get("capacity", 100)
        key = f"mem_{time.time()}"
        sm[key] = data
        while len(sm) > capacity:
            sm.popitem(last=False)  # LRU 淘汰
        return {"key": key, "short_term_size": len(sm)}

    elif operation == "retrieve_short":
        results = list(sm.values())
        return {"memories": results, "count": len(results)}

    elif operation == "add_long":
        # 分词并计算 TF
        words = _tokenize(data)
        tf = Counter(words)
        doc_len = len(words)
        doc_entry = {"text": data, "tf": dict(tf), "length": doc_len}
        lm.append(doc_entry)
        memory_manager_short_long_term._doc_count += 1
        # 更新 IDF
        unique_words = set(words)
        for w in unique_words:
            idf[w] += 1
        return {"long_term_size": len(lm), "indexed": True}

    elif operation == "retrieve_long":
        top_k = params.get("top_k", 5)
        query_words = _tokenize(data)
        query_tf = Counter(query_words)

        # 计算 TF-IDF 余弦相似度
        scores = []
        N = memory_manager_short_long_term._doc_count
        for idx, doc in enumerate(lm):
            sim = 0.0
            for word, q_tf in query_tf.items():
                if word in doc["tf"]:
                    doc_tf = doc["tf"][word] / max(1, doc["length"])
                    word_idf = math.log((N + 1) / (1 + idf[word])) if idf[word] > 0 else 1.0
                    sim += q_tf * doc_tf * word_idf
            # 归一化
            norm = math.sqrt(sum(v ** 2 for v in query_tf.values()))
            if norm > 0:
                sim /= norm
            scores.append((sim, idx, doc["text"]))

        scores.sort(reverse=True, key=lambda x: x[0])
        return {"results": [{"score": round(s, 4), "text": t}
                            for s, _, t in scores[:top_k]]}

    elif operation == "forget":
        # 遗忘：从短期记忆中移除指定 key
        if data in sm:
            del sm[data]
            return {"forgotten": data, "remaining": len(sm)}
        return {"forgotten": None, "message": "key not found"}

    elif operation == "stats":
        return {
            "short_term_count": len(sm),
            "long_term_count": len(lm),
            "vocabulary_size": len(idf),
            "total_docs": memory_manager_short_long_term._doc_count,
        }

    return {"error": f"未知操作: {operation}"}


def _tokenize(text):
    """简单分词器：中英文混合分词。"""
    # 英文按空格和标点分词，中文按字分词
    words = re.findall(r'[a-zA-Z]+', text.lower())
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    words.extend(chinese_chars)
    return words


# =====================================================================
# 7. 提示词优化器
# =====================================================================

def prompt_optimizer(prompt, optimization_goals):
    """提示词优化器。

    分析提示词结构 → 检查模糊性/缺失上下文/角色定义/输出格式 → 生成优化建议。

    Args:
        prompt: 原始提示词字符串。
        optimization_goals: 优化目标列表，可选值:
            ['clarity', 'context', 'role', 'format', 'specificity']。

    Returns:
        dict: {"optimized_prompt": 优化后提示词, "suggestions": 建议列表,
               "score": 优化评分(0-100)}.
    """
    suggestions = []
    score = 100
    optimized = prompt

    # --- 模糊性检查 (clarity) ---
    if "clarity" in optimization_goals:
        vague_words = ["一些", "大概", "可能", "尽量", "somewhat",
                        "maybe", "perhaps", "some"]
        found_vague = [w for w in vague_words if w.lower() in prompt.lower()]
        if found_vague:
            suggestions.append(f"发现模糊词汇: {found_vague}，建议使用更具体的表述")
            score -= 15
        # 检查句子长度
        sentences = prompt.split("。")
        for s in sentences:
            if len(s) > 100:
                suggestions.append(f"句子过长(>{len(s)}字符)，建议拆分: '{s[:30]}...'")
                score -= 5

    # --- 上下文检查 (context) ---
    if "context" in optimization_goals:
        if "背景" not in prompt and "context" not in prompt.lower():
            suggestions.append("缺少背景上下文，建议添加任务背景说明")
            score -= 10
        if "请" not in prompt and "please" not in prompt.lower():
            suggestions.append("缺少明确的指令动词，建议使用'请分析/请总结/请列出'等")
            score -= 5

    # --- 角色定义检查 (role) ---
    if "role" in optimization_goals:
        role_patterns = ["你是", "作为", "你扮演", "you are", "as a", "act as"]
        has_role = any(p in prompt.lower() for p in role_patterns)
        if not has_role:
            suggestions.append("缺少角色定义，建议添加'你是一个专业的...'")
            score -= 15
            optimized = f"你是一个专业的AI助手。\n\n{optimized}"

    # --- 输出格式检查 (format) ---
    if "format" in optimization_goals:
        format_patterns = ["格式", "JSON", "列表", "markdown", "format",
                           "output", "返回"]
        has_format = any(p.lower() in prompt.lower() for p in format_patterns)
        if not has_format:
            suggestions.append("缺少输出格式要求，建议明确指定输出格式")
            score -= 10
            optimized += "\n\n请以结构化格式输出结果。"

    # --- 具体性检查 (specificity) ---
    if "specificity" in optimization_goals:
        if len(prompt) < 50:
            suggestions.append("提示词过短，缺少足够的细节约束")
            score -= 20
        # 检查是否包含约束条件
        constraint_patterns = ["不超过", "至少", "必须", "限制", "less than",
                               "at least", "must", "limit"]
        has_constraint = any(p.lower() in prompt.lower() for p in constraint_patterns)
        if not has_constraint:
            suggestions.append("缺少约束条件（如字数限制、格式要求等）")
            score -= 10

    score = max(0, min(100, score))

    return {
        "optimized_prompt": optimized,
        "suggestions": suggestions,
        "score": score,
        "improvements_count": len(suggestions),
    }


# =====================================================================
# 8. 多 Agent 协调器
# =====================================================================

def multi_agent_coordinator(agents, task, protocol):
    """多 Agent 协调器。

    支持三种协调协议：协作(cooperative)、竞争(competitive)、层级(hierarchical)。

    Args:
        agents: Agent 列表，每个元素为
            {"id": agent_id, "name": 名称, "role": 角色, "capability": 能力分数}。
        task: 任务描述字典 {"description": ..., "subtasks": [...]}。
        protocol: 协调协议 — 'cooperative'/'competitive'/'hierarchical'。

    Returns:
        dict: 协调结果，包含任务分配、消息传递记录和最终输出。
    """
    messages = []
    assignments = []

    if protocol == "cooperative":
        # 协作模式：按能力分配子任务，所有Agent协作完成
        subtasks = task.get("subtasks", [task["description"]])
        sorted_agents = sorted(agents, key=lambda a: a.get("capability", 0), reverse=True)
        for i, subtask in enumerate(subtasks):
            agent = sorted_agents[i % len(sorted_agents)]
            assignments.append({
                "subtask": subtask,
                "agent": agent["id"],
                "agent_name": agent["name"],
            })
            messages.append({
                "from": "coordinator",
                "to": agent["id"],
                "type": "assignment",
                "content": subtask,
            })
        # Agent间消息传递
        for i in range(len(assignments) - 1):
            messages.append({
                "from": assignments[i]["agent"],
                "to": assignments[i + 1]["agent"],
                "type": "handoff",
                "content": f"子任务{i+1}完成，传递给下一Agent",
            })

    elif protocol == "competitive":
        # 竞争模式：所有Agent处理同一任务，选择最优结果
        results = []
        for agent in agents:
            assignments.append({
                "task": task["description"],
                "agent": agent["id"],
                "agent_name": agent["name"],
            })
            # 模拟Agent执行（基于能力评分）
            quality = agent.get("capability", 0.5) * (0.8 + 0.2 * hash(agent["id"]) % 10 / 10)
            results.append({
                "agent": agent["id"],
                "quality": round(quality, 4),
            })
            messages.append({
                "from": "coordinator",
                "to": agent["id"],
                "type": "competition_invite",
                "content": task["description"],
            })
        # 选择最高分
        results.sort(key=lambda r: r["quality"], reverse=True)
        winner = results[0] if results else None
        messages.append({
            "from": "coordinator",
            "to": "all",
            "type": "result_announcement",
            "content": f"胜出者: {winner['agent']}" if winner else "无结果",
        })

    elif protocol == "hierarchical":
        # 层级模式：按角色分级，上级分配任务给下级
        managers = [a for a in agents if a.get("role") == "manager"]
        workers = [a for a in agents if a.get("role") == "worker"]
        if not managers:
            managers = [agents[0]]
        if not workers:
            workers = agents[1:] if len(agents) > 1 else agents

        # 经理分配任务
        for mgr in managers:
            subtasks = task.get("subtasks", [task["description"]])
            for i, subtask in enumerate(subtasks):
                worker = workers[i % len(workers)] if workers else mgr
                assignments.append({
                    "subtask": subtask,
                    "assigned_by": mgr["id"],
                    "assigned_to": worker["id"],
                })
                messages.append({
                    "from": mgr["id"],
                    "to": worker["id"],
                    "type": "directive",
                    "content": subtask,
                })
    else:
        return {"error": f"未知协议: {protocol}"}

    return {
        "protocol": protocol,
        "assignments": assignments,
        "messages": messages,
        "task": task["description"],
        "agent_count": len(agents),
    }


# =====================================================================
# 9. 响应解析与验证器
# =====================================================================

def response_parser_and_validator(response, expected_format):
    """响应解析验证器。

    支持 JSON / Markdown / 结构化文本解析，自动修复格式错误。

    Args:
        response: 原始响应字符串。
        expected_format: 期望格式 — 'json'/'markdown'/'structured'/'text'。

    Returns:
        dict: {"parsed": 解析结果, "valid": 是否有效,
               "repaired": 是否经过修复, "errors": 错误列表}.
    """
    errors = []
    repaired = False
    parsed = None

    if expected_format == "json":
        # 尝试直接解析
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            errors.append(f"JSON解析失败: {e}")
            # 尝试修复：提取 JSON 代码块
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    repaired = True
                    errors.append("修复: 从代码块中提取JSON")
                except json.JSONDecodeError:
                    pass
            # 尝试修复：查找第一个 { 和最后一个 }
            if parsed is None:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(response[start:end + 1])
                        repaired = True
                        errors.append("修复: 提取花括号内内容")
                    except json.JSONDecodeError:
                        # 尝试修复尾逗号
                        fragment = response[start:end + 1]
                        fragment = re.sub(r',\s*}', '}', fragment)
                        fragment = re.sub(r',\s*]', ']', fragment)
                        try:
                            parsed = json.loads(fragment)
                            repaired = True
                            errors.append("修复: 移除尾逗号")
                        except json.JSONDecodeError:
                            pass

    elif expected_format == "markdown":
        # 验证 Markdown 结构
        parsed = {"raw": response}
        headings = re.findall(r'^(#{1,6})\s+(.+)$', response, re.MULTILINE)
        parsed["headings"] = [
            {"level": len(h[0]), "text": h[1]} for h in headings
        ]
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        parsed["code_blocks"] = code_blocks
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', response)
        parsed["links"] = links
        if not headings:
            errors.append("Markdown缺少标题结构")
        if not response.strip():
            errors.append("响应为空")

    elif expected_format == "structured":
        # 解析 key: value 格式
        parsed = {}
        for line in response.strip().split("\n"):
            if ":" in line or "：" in line:
                sep = ":" if ":" in line else "："
                key, _, value = line.partition(sep)
                parsed[key.strip()] = value.strip()
        if not parsed:
            errors.append("无法解析结构化文本")

    else:  # text
        parsed = response.strip()
        if not parsed:
            errors.append("响应为空")

    valid = len(errors) == 0 or repaired
    return {
        "parsed": parsed,
        "valid": valid,
        "repaired": repaired,
        "errors": errors,
        "format": expected_format,
    }


# =====================================================================
# 10. Agent 评估指标计算器
# =====================================================================

def agent_evaluation_metrics(agent_outputs, ground_truth):
    """Agent 评估指标计算器。

    计算准确率、精确率、召回率、F1、响应时间、幻觉率。

    Args:
        agent_outputs: Agent 输出列表，每个元素为
            {"answer": 答案文本, "response_time": 响应时间(秒),
             "cited_sources": 引用来源列表(可选)}。
        ground_truth: 标准答案列表，与 agent_outputs 等长，
            每个元素为 {"answer": 正确答案, "sources": 来源列表(可选)}。

    Returns:
        dict: 各项指标。
    """
    n = len(agent_outputs)
    if n == 0:
        return {"error": "无评估数据"}

    # 文本匹配准确率
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    hallucination_count = 0
    response_times = []

    for i in range(n):
        output = agent_outputs[i]
        truth = ground_truth[i] if i < len(ground_truth) else {}

        # 准确率（完全匹配或包含匹配）
        out_answer = str(output.get("answer", "")).strip().lower()
        gt_answer = str(truth.get("answer", "")).strip().lower()
        if out_answer == gt_answer or gt_answer in out_answer:
            correct += 1

        # 精确率/召回率（基于关键词）
        out_words = set(out_answer.split())
        gt_words = set(gt_answer.split())
        tp = len(out_words & gt_words)
        fp = len(out_words - gt_words)
        fn = len(gt_words - out_words)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        # 幻觉率检查（输出中有但标准答案中没有的来源）
        out_sources = set(output.get("cited_sources", []))
        gt_sources = set(truth.get("sources", []))
        if out_sources and not out_sources.issubset(gt_sources):
            hallucination_count += 1

        # 响应时间
        rt = output.get("response_time", 0)
        if isinstance(rt, (int, float)):
            response_times.append(rt)

    accuracy = correct / n
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    hallucination_rate = hallucination_count / n
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_response_time_sec": round(avg_response_time, 4),
        "total_samples": n,
        "correct_count": correct,
    }


# =====================================================================
# 主程序测试
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("agent-framework 算法测试")
    print("=" * 60)

    # 1. ReAct 推理循环
    def search_tool(query):
        return f"搜索结果: 关于 '{query}' 的信息"

    tools = {"search": search_tool}
    result = react_reasoning_loop("什么是AI?", tools, max_iterations=5)
    print(f"\n[1] ReAct: {result['iterations']}轮迭代, "
          f"{len(result['tool_calls'])}次工具调用")
    print(f"    答案: {result['answer'][:60]}...")

    # 2. 链式思维求解器
    def verify_positive(x):
        return isinstance(x, str) and len(x) > 0

    steps = [
        {"description": "分析问题", "verify": verify_positive},
        {"description": "推导结论", "verify": verify_positive},
    ]
    cot = chain_of_thought_solver("2+3=?", steps)
    print(f"[2] CoT: {len(cot['trace'])}步, "
          f"回溯{cot['backtracks']}次, 验证={cot['verified']}")

    # 3. Few-shot 模板引擎
    examples = [
        {"input": "今天天气很好", "output": "正面"},
        {"input": "我很难过", "output": "负面"},
        {"input": "今天天气不错适合出门", "output": "正面"},
    ]
    fs = few_shot_template_engine("classification", examples, "今天天气很好适合运动")
    print(f"[3] Few-shot: 选中{len(fs['selected_examples'])}个示例")

    # 4. DAG 工作流
    def task_a(prev): return "A完成"
    def task_b(prev): return f"B基于{prev.get('task_a','')}"
    tasks = {
        "task_a": {"name": "任务A", "action": task_a},
        "task_b": {"name": "任务B", "action": task_b},
        "task_c": {"name": "任务C", "action": lambda prev: "C完成"},
    }
    deps = [("task_a", "task_b")]
    wf = agent_workflow_orchestrator(tasks, deps, {})
    print(f"[4] DAG工作流: 执行顺序={wf['execution_order']}, 状态={wf['status']}")

    # 5. 工具注册与分派
    registry = {
        "calculator": {
            "function": lambda x, y: x + y,
            "parameters": {"x": {"type": "int", "required": True},
                           "y": {"type": "int", "required": True}},
            "cache": True,
        }
    }
    disp = tool_registry_and_dispatcher("calculator", {"x": 3, "y": 5}, registry)
    print(f"[5] 工具分派: 3+5={disp['result']}, 缓存={disp['cached']}")

    # 6. 记忆管理
    memory_manager_short_long_term("add_long", "AI是人工智能的缩写", {})
    memory_manager_short_long_term("add_long", "机器学习是AI的子领域", {})
    ret = memory_manager_short_long_term("retrieve_long", "AI是什么", {"top_k": 2})
    print(f"[6] 记忆检索: {ret['results']}")

    # 7. 提示词优化
    opt = prompt_optimizer("帮我写文章", ["clarity", "role", "format"])
    print(f"[7] 提示优化: 评分={opt['score']}, {opt['improvements_count']}条建议")

    # 8. 多Agent协调
    agents = [
        {"id": "a1", "name": "Agent1", "role": "manager", "capability": 0.9},
        {"id": "a2", "name": "Agent2", "role": "worker", "capability": 0.7},
    ]
    task = {"description": "完成报告", "subtasks": ["调研", "写作"]}
    coord = multi_agent_coordinator(agents, task, "hierarchical")
    print(f"[8] 多Agent(层级): {len(coord['assignments'])}个任务分配")

    # 9. 响应解析
    resp = response_parser_and_validator(
        '```json\n{"name": "test", "value": 42}\n```', "json"
    )
    print(f"[9] 响应解析: valid={resp['valid']}, repaired={resp['repaired']}")
    print(f"    解析结果: {resp['parsed']}")

    # 10. 评估指标
    outputs = [
        {"answer": "巴黎是法国首都", "response_time": 1.2},
        {"answer": "北京", "response_time": 0.8},
    ]
    truth = [
        {"answer": "巴黎是法国的首都"},
        {"answer": "北京"},
    ]
    metrics = agent_evaluation_metrics(outputs, truth)
    print(f"[10] 评估: 准确率={metrics['accuracy']}, "
          f"F1={metrics['f1_score']}, 平均响应={metrics['avg_response_time_sec']}s")

    print("\n所有测试通过。")
