"""AI Agent 框架工具集。

提供10个Agent开发常用功能，包括提示词模板管理、链式思维求解、
Few-shot分类、工作流构建、响应解析、Token计数、对话摘要、
提示优化、多Agent协调和评估指标计算。
"""

import json
import math
import re
import hashlib
import textwrap
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# 模块级存储（模拟持久化层）
# ---------------------------------------------------------------------------
_TEMPLATE_STORE = {}
_WORKFLOW_STORE = {}
_VERSION_STORE = defaultdict(list)


# ---------------------------------------------------------------------------
# 1. prompt_template_manager
# ---------------------------------------------------------------------------
def prompt_template_manager(action, template_name, template_content="", variables=None):
    """提示词模板管理（增删改查）。

    Args:
        action (str): 操作类型，可选值: 'create', 'read', 'update', 'delete', 'list'。
        template_name (str): 模板名称（唯一标识）。
        template_content (str, optional): 模板内容，支持 {variable} 占位符。默认为空。
        variables (dict, optional): 变量键值对，用于填充模板占位符。默认为 None。

    Returns:
        dict: 操作结果，包含 'success', 'action', 'name' 及相应数据。

    Raises:
        ValueError: 不支持的操作或模板不存在时抛出。
    """
    if action == "create":
        if template_name in _TEMPLATE_STORE:
            raise ValueError(f"模板 '{template_name}' 已存在")
        _TEMPLATE_STORE[template_name] = {
            "content": template_content,
            "variables": list(variables.keys()) if variables else [],
        }
        return {"success": True, "action": "create", "name": template_name}

    elif action == "read":
        if template_name not in _TEMPLATE_STORE:
            raise ValueError(f"模板 '{template_name}' 不存在")
        tmpl = _TEMPLATE_STORE[template_name]
        content = tmpl["content"]
        if variables:
            try:
                content = content.format(**variables)
            except KeyError as e:
                raise ValueError(f"缺少变量: {e}")
        return {"success": True, "action": "read", "name": template_name, "content": content}

    elif action == "update":
        if template_name not in _TEMPLATE_STORE:
            raise ValueError(f"模板 '{template_name}' 不存在")
        _TEMPLATE_STORE[template_name]["content"] = template_content
        if variables:
            _TEMPLATE_STORE[template_name]["variables"] = list(variables.keys())
        return {"success": True, "action": "update", "name": template_name}

    elif action == "delete":
        if template_name not in _TEMPLATE_STORE:
            raise ValueError(f"模板 '{template_name}' 不存在")
        del _TEMPLATE_STORE[template_name]
        return {"success": True, "action": "delete", "name": template_name}

    elif action == "list":
        return {
            "success": True,
            "action": "list",
            "templates": list(_TEMPLATE_STORE.keys()),
        }

    else:
        raise ValueError(f"不支持的操作: '{action}'，可选: create, read, update, delete, list")


# ---------------------------------------------------------------------------
# 2. chain_of_thought_solver
# ---------------------------------------------------------------------------
def chain_of_thought_solver(problem, steps=None):
    """链式思维求解器，将问题分解为步骤逐步推理。

    Args:
        problem (str): 问题描述。
        steps (list, optional): 预定义步骤列表。如果为None则自动分解。默认为 None。

    Returns:
        dict: 包含 'problem', 'steps', 'reasoning', 'conclusion' 的字典。
    """
    if steps is None:
        steps = [
            "理解问题：分析题目要求和已知条件",
            "识别关键信息：提取问题中的核心要素",
            "制定策略：确定解决问题的方法路径",
            "执行推理：按策略逐步推导",
            "验证结果：检查推理过程和结论的一致性",
            "得出结论：总结最终答案",
        ]

    reasoning_parts = []
    for i, step in enumerate(steps, 1):
        reasoning_parts.append(f"步骤 {i}: {step}")
        reasoning_parts.append(f"  -> 针对 '{problem}' 的分析: 正在执行 {step[:30]}...")

    conclusion = (
        f"经过 {len(steps)} 步推理，问题 '{problem[:50]}...' 已分析完成。"
        if len(problem) > 50
        else f"经过 {len(steps)} 步推理，问题 '{problem}' 已分析完成。"
    )

    return {
        "problem": problem,
        "steps": steps,
        "reasoning": "\n".join(reasoning_parts),
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# 3. few_shot_classifier
# ---------------------------------------------------------------------------
def few_shot_classifier(input_text, examples, labels):
    """Few-shot 分类器，基于示例对输入文本进行分类。

    Args:
        input_text (str): 待分类的输入文本。
        examples (list): 示例列表，每个元素为 {"text": str, "label": str}。
        labels (list): 所有可能的标签列表。

    Returns:
        dict: 包含 'input', 'predicted_label', 'confidence', 'scores' 的字典。
    """
    if not examples:
        raise ValueError("examples 不能为空")
    if not labels:
        raise ValueError("labels 不能为空")

    input_words = set(input_text.lower().split())
    scores = {}

    for label in labels:
        label_examples = [ex for ex in examples if ex["label"] == label]
        if not label_examples:
            scores[label] = 0.0
            continue
        total_overlap = 0
        for ex in label_examples:
            ex_words = set(ex["text"].lower().split())
            overlap = len(input_words & ex_words)
            total_overlap += overlap
        scores[label] = total_overlap / len(label_examples)

    max_score = max(scores.values()) if scores else 0
    predicted = max(scores, key=scores.get) if scores else labels[0]
    confidence = max_score / sum(scores.values()) if sum(scores.values()) > 0 else 0.0

    return {
        "input": input_text,
        "predicted_label": predicted,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 4. agent_workflow_builder
# ---------------------------------------------------------------------------
def agent_workflow_builder(steps, inputs):
    """Agent工作流构建器，构建有向无环图（DAG）工作流。

    Args:
        steps (list): 步骤列表，每个元素为 dict:
            {"id": str, "name": str, "depends_on": [str], "handler": str}。
        inputs (dict): 输入数据键值对。

    Returns:
        dict: 包含 'dag', 'execution_order', 'inputs' 的工作流定义。

    Raises:
        ValueError: 存在循环依赖或缺少依赖时抛出。
    """
    step_map = {s["id"]: s for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            if dep not in step_map:
                raise ValueError(f"步骤 '{s['id']}' 依赖不存在的步骤 '{dep}'")

    # 拓扑排序
    in_degree = {s["id"]: 0 for s in steps}
    adj = defaultdict(list)
    for s in steps:
        for dep in s.get("depends_on", []):
            adj[dep].append(s["id"])
            in_degree[s["id"]] += 1

    queue = deque([sid for sid, d in in_degree.items() if d == 0])
    execution_order = []
    while queue:
        sid = queue.popleft()
        execution_order.append(sid)
        for neighbor in adj[sid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(execution_order) != len(steps):
        raise ValueError("工作流中存在循环依赖")

    return {
        "dag": {
            "nodes": [s["id"] for s in steps],
            "edges": [
                {"from": dep, "to": s["id"]}
                for s in steps
                for dep in s.get("depends_on", [])
            ],
        },
        "execution_order": execution_order,
        "inputs": inputs,
        "steps": {s["id"]: {"name": s["name"], "handler": s.get("handler", "")} for s in steps},
    }


# ---------------------------------------------------------------------------
# 5. response_parser
# ---------------------------------------------------------------------------
def response_parser(response, format_type):
    """LLM响应解析器，支持JSON/Markdown/XML格式。

    Args:
        response (str): LLM返回的原始文本。
        format_type (str): 期望的格式类型: 'json', 'markdown', 'xml'。

    Returns:
        dict: 解析结果，包含 'format', 'parsed', 'raw' 字段。

    Raises:
        ValueError: 不支持的格式或解析失败时抛出。
    """
    format_type = format_type.lower().strip()

    if format_type == "json":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError("无法从响应中提取JSON")
        return {"format": "json", "parsed": parsed, "raw": response}

    elif format_type == "markdown":
        sections = []
        current_header = None
        current_content = []
        for line in response.split("\n"):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                if current_header:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_content),
                    })
                current_header = header_match.group(2)
                current_content = []
            else:
                current_content.append(line)
        if current_header:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_content),
            })
        return {"format": "markdown", "parsed": sections, "raw": response}

    elif format_type == "xml":
        elements = {}
        pattern = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
        for match in pattern.finditer(response):
            elements[match.group(1)] = match.group(2).strip()
        return {"format": "xml", "parsed": elements, "raw": response}

    else:
        raise ValueError(f"不支持的格式: '{format_type}'，可选: json, markdown, xml")


# ---------------------------------------------------------------------------
# 6. token_counter
# ---------------------------------------------------------------------------
def token_counter(text, model="gpt-3.5-turbo"):
    """Token计数器，估算文本的token数量。

    Args:
        text (str): 输入文本。
        model (str, optional): 模型名称。不同模型使用不同的计数系数。默认为 'gpt-3.5-turbo'。

    Returns:
        dict: 包含 'text', 'model', 'char_count', 'word_count', 'estimated_tokens' 的字典。
    """
    model_coefficients = {
        "gpt-3.5-turbo": 0.75,
        "gpt-4": 0.75,
        "gpt-4o": 0.70,
        "claude": 0.80,
        "gemini": 0.72,
        "default": 0.75,
    }
    coeff = model_coefficients.get(model, model_coefficients["default"])

    char_count = len(text)
    word_count = len(text.split())

    # 英文单词约 1.3 token/word，中文约 2 token/char
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = char_count - chinese_chars

    estimated_tokens = int(chinese_chars * 2 + english_chars * coeff)
    if estimated_tokens == 0:
        estimated_tokens = max(1, int(char_count * coeff))

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "model": model,
        "char_count": char_count,
        "word_count": word_count,
        "chinese_chars": chinese_chars,
        "estimated_tokens": estimated_tokens,
    }


# ---------------------------------------------------------------------------
# 7. conversation_summarizer
# ---------------------------------------------------------------------------
def conversation_summarizer(messages):
    """对话摘要生成器。

    Args:
        messages (list): 消息列表，每个元素为 {"role": str, "content": str}。

    Returns:
        dict: 包含 'summary', 'message_count', 'key_points', 'participants' 的字典。

    Raises:
        ValueError: messages 为空时抛出。
    """
    if not messages:
        raise ValueError("messages 不能为空")

    participants = list(set(m["role"] for m in messages))
    key_points = []
    total_chars = 0

    for msg in messages:
        content = msg["content"]
        total_chars += len(content)
        sentences = re.split(r'[。.!?！？\n]', content)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                key_points.append(f"[{msg['role']}] {sent[:80]}")

    avg_length = total_chars // len(messages)
    summary = (
        f"本次对话共 {len(messages)} 条消息，参与者: {', '.join(participants)}。"
        f"平均每条消息 {avg_length} 字符。"
        f"主要讨论了 {min(5, len(key_points))} 个要点。"
    )

    return {
        "summary": summary,
        "message_count": len(messages),
        "participants": participants,
        "key_points": key_points[:10],
        "total_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# 8. prompt_optimizer
# ---------------------------------------------------------------------------
def prompt_optimizer(original_prompt, target="clarity"):
    """提示词优化器。

    Args:
        original_prompt (str): 原始提示词。
        target (str, optional): 优化目标，可选: 'clarity'(清晰度),
            'specificity'(具体性), 'structure'(结构化), 'completeness'(完整性)。
            默认为 'clarity'。

    Returns:
        dict: 包含 'original', 'optimized', 'target', 'improvements' 的字典。
    """
    improvements = []
    optimized = original_prompt

    if target == "clarity":
        if len(original_prompt) < 20:
            improvements.append("提示词过短，建议添加更多上下文")
            optimized = f"请详细描述以下任务: {original_prompt}。请提供具体的要求和期望的输出格式。"
        if "?" not in original_prompt and len(original_prompt) > 50:
            improvements.append("建议以明确的问句或指令结尾")
            optimized += "\n\n请按照以上要求执行。"

    elif target == "specificity":
        improvements.append("添加具体的输出格式要求")
        improvements.append("指定角色和语境")
        optimized = (
            f"你是一个专业助手。\n\n任务: {original_prompt}\n\n"
            f"要求:\n1. 输出格式为结构化文本\n2. 包含具体步骤\n3. 提供示例\n"
        )

    elif target == "structure":
        improvements.append("将提示词拆分为多个结构化部分")
        optimized = (
            f"## 角色\n专业助手\n\n"
            f"## 任务\n{original_prompt}\n\n"
            f"## 约束\n- 输出简洁明了\n- 逻辑清晰\n\n"
            f"## 输出格式\n结构化文本"
        )

    elif target == "completeness":
        improvements.append("添加示例说明期望输出")
        improvements.append("指定边界条件和异常处理")
        optimized = (
            f"{original_prompt}\n\n"
            f"请考虑以下方面:\n"
            f"- 边界情况处理\n- 错误处理建议\n- 输出验证标准\n"
            f"- 示例: 输入 -> 期望输出"
        )

    else:
        raise ValueError(f"不支持的优化目标: '{target}'")

    return {
        "original": original_prompt,
        "optimized": optimized,
        "target": target,
        "improvements": improvements,
        "original_length": len(original_prompt),
        "optimized_length": len(optimized),
    }


# ---------------------------------------------------------------------------
# 9. multi_agent_coordinator
# ---------------------------------------------------------------------------
def multi_agent_coordinator(agents_config, task):
    """多Agent协调器，分配任务给多个Agent并汇总结果。

    Args:
        agents_config (list): Agent配置列表，每个元素为:
            {"id": str, "role": str, "capabilities": [str]}。
        task (str): 需要协调完成的主任务。

    Returns:
        dict: 包含 'task', 'assignments', 'execution_plan', 'agents' 的协调方案。
    """
    if not agents_config:
        raise ValueError("agents_config 不能为空")

    # 任务分解
    subtasks = [
        {"subtask": f"分析任务: {task[:40]}", "required_capability": "analysis"},
        {"subtask": f"执行核心逻辑: {task[:40]}", "required_capability": "execution"},
        {"subtask": "验证和汇总结果", "required_capability": "verification"},
    ]

    assignments = []
    for subtask in subtasks:
        best_agent = None
        best_score = -1
        for agent in agents_config:
            score = sum(
                1 for cap in agent.get("capabilities", [])
                if cap in subtask["required_capability"]
            )
            if score > best_score:
                best_score = score
                best_agent = agent
        if best_agent is None:
            best_agent = agents_config[0]
        assignments.append({
            "agent_id": best_agent["id"],
            "agent_role": best_agent["role"],
            "subtask": subtask["subtask"],
            "required_capability": subtask["required_capability"],
        })

    execution_plan = []
    for i, assignment in enumerate(assignments, 1):
        execution_plan.append(f"阶段{i}: Agent '{assignment['agent_id']}' ({assignment['agent_role']}) -> {assignment['subtask']}")

    return {
        "task": task,
        "agents": [{"id": a["id"], "role": a["role"], "capabilities": a.get("capabilities", [])} for a in agents_config],
        "assignments": assignments,
        "execution_plan": execution_plan,
    }


# ---------------------------------------------------------------------------
# 10. eval_metric_calculator
# ---------------------------------------------------------------------------
def eval_metric_calculator(predictions, ground_truth, metric="accuracy"):
    """评估指标计算器。

    Args:
        predictions (list): 预测值列表。
        ground_truth (list): 真实值列表。
        metric (str, optional): 评估指标，可选: 'accuracy', 'precision', 'recall',
            'f1', 'mse', 'mae'。默认为 'accuracy'。

    Returns:
        dict: 包含 'metric', 'value', 'details' 的评估结果。

    Raises:
        ValueError: 长度不匹配或不支持的指标时抛出。
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions({len(predictions)}) 和 ground_truth({len(ground_truth)}) 长度不匹配"
        )
    if not predictions:
        raise ValueError("预测列表不能为空")

    n = len(predictions)

    if metric == "accuracy":
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        value = correct / n
        return {"metric": "accuracy", "value": round(value, 4), "details": {"correct": correct, "total": n}}

    elif metric in ("precision", "recall", "f1"):
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if metric == "precision":
            return {"metric": "precision", "value": round(precision, 4), "details": {"tp": tp, "fp": fp}}
        elif metric == "recall":
            return {"metric": "recall", "value": round(recall, 4), "details": {"tp": tp, "fn": fn}}
        else:  # f1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return {"metric": "f1", "value": round(f1, 4), "details": {"precision": round(precision, 4), "recall": round(recall, 4)}}

    elif metric == "mse":
        mse = sum((p - g) ** 2 for p, g in zip(predictions, ground_truth)) / n
        return {"metric": "mse", "value": round(mse, 4), "details": {"n": n}}

    elif metric == "mae":
        mae = sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / n
        return {"metric": "mae", "value": round(mae, 4), "details": {"n": n}}

    else:
        raise ValueError(
            f"不支持的指标: '{metric}'，可选: accuracy, precision, recall, f1, mse, mae"
        )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(textwrap.dedent("""\
        agent-framework 模块已加载。
        可用函数:
          - prompt_template_manager
          - chain_of_thought_solver
          - few_shot_classifier
          - agent_workflow_builder
          - response_parser
          - token_counter
          - conversation_summarizer
          - prompt_optimizer
          - multi_agent_coordinator
          - eval_metric_calculator
    """))
