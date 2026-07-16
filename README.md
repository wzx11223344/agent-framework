# agent-framework

[![CI](https://github.com/wzx11223344/agent-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/wzx11223344/agent-framework/actions/workflows/ci.yml)

AI Agent 框架工具集 — 纯 Python 标准库实现，无外部依赖。

## 功能列表

| # | 函数 | 说明 | 核心算法 |
|---|------|------|----------|
| 1 | `react_reasoning_loop` | ReAct推理循环 | Thought→Action→Observation循环 + 推理链追踪 |
| 2 | `chain_of_thought_solver` | 链式思维求解器 | 分步推理 + 中间结果验证 + 回溯 |
| 3 | `few_shot_template_engine` | Few-shot模板引擎 | Jaccard相似度动态选择示例 + 模板填充 |
| 4 | `agent_workflow_orchestrator` | 工作流编排器 | DAG依赖图 + Kahn拓扑排序 + 并行/串行/条件分支 |
| 5 | `tool_registry_and_dispatcher` | 工具注册分派 | 签名验证 + 参数校验 + MD5结果缓存 |
| 6 | `memory_manager_short_long_term` | 记忆管理 | LRU短期记忆 + TF-IDF长期记忆检索 |
| 7 | `prompt_optimizer` | 提示词优化器 | 结构分析 + 模糊性检测 + 上下文缺失检查 |
| 8 | `multi_agent_coordinator` | 多Agent协调器 | 协作/竞争/层级三种协议 + 消息传递 |
| 9 | `response_parser_and_validator` | 响应解析验证器 | JSON/Markdown解析 + 自动修复格式错误 |
| 10 | `agent_evaluation_metrics` | Agent评估指标 | 准确率/精确率/召回率/F1/幻觉率 |

## 算法原理与复杂度分析

### 1. ReAct推理循环
- **原理**: 实现Reasoning+Acting范式 — 每轮先生成Thought(推理过程)，然后选择Action(工具调用)，获取Observation(工具返回)，循环直到得出最终答案或达到最大迭代。支持推理链追踪记录完整执行路径。内置工具调用计数安全检查，防止无限循环。
- **时间复杂度**: O(I×T)，I=最大迭代数，T=每次工具调用时间。

### 2. 链式思维求解器
- **原理**: 将问题分解为多步推理 — 按步骤列表逐步执行，每步生成中间结果并验证正确性；若验证失败则回溯到上一步重新推理。支持数学计算、逻辑推理、文本推理三类问题。
- **时间复杂度**: O(S×V)，S=步骤数，V=每步验证开销。

### 3. Few-shot模板引擎
- **原理**: 从示例库中通过Jaccard相似度 `J(A,B) = |A∩B| / |A∪B|` 动态选择与查询最相关的Top-K示例，按相似度排序后填充模板。避免固定示例的选择偏差。
- **时间复杂度**: O(N×|q|)，N=示例数，|q|=查询词项数。

### 4. Agent工作流编排器
- **原理**: 构建DAG(有向无环图)任务依赖图 → Kahn算法拓扑排序(入度归零法) → 按依赖顺序调度Agent执行。支持并行调度(同层无依赖任务并行)、串行执行、条件分支(根据上一步结果选择路径)。
- **时间复杂度**: O(V+E)，V=任务数，E=依赖边数（Kahn算法）。

### 5. 工具注册与分派
- **原理**: 工具注册时记录函数签名(参数名/类型/默认值)；调用时进行参数类型校验和必填检查；结果缓存用MD5哈希 `(tool_name + args)` 作为key，命中则直接返回缓存。
- **时间复杂度**: O(1) 工具查找，O(1) 缓存命中。

### 6. 记忆管理系统
- **原理**: 短期记忆使用LRU(最近最少使用)缓存，容量满时淘汰最久未访问项；长期记忆使用TF-IDF索引，查询时计算查询向量与记忆向量的余弦相似度 `cos(θ) = (A·B)/(|A|×|B|)`，返回Top-K最相关记忆。
- **时间复杂度**: 短期记忆O(1)读写；长期检索O(N×D)，N=记忆数，D=向量维度。

### 7. 提示词优化器
- **原理**: 多维度分析提示词 — 检查角色定义是否清晰、输出格式是否指定、上下文是否充分、指令是否模糊(如"好一点"等模糊词)、示例是否充足。生成针对性优化建议。
- **时间复杂度**: O(L)，L=提示词长度。

### 8. 多Agent协调器
- **原理**: 支持三种协调协议 — 协作: 所有Agent共同完成任务，结果合并；竞争: Agent独立完成，选最优结果；层级: 管理者Agent分配子任务给工作者Agent，汇总结果。
- **时间复杂度**: O(A×M)，A=Agent数，M=消息轮数。

### 9. 响应解析验证器
- **原理**: 自动检测响应格式(JSON/Markdown/结构化文本) → 提取内容 → 验证字段完整性 → 若JSON格式错误(如尾随逗号、缺引号)尝试自动修复(正则替换+json.loads重试)。
- **时间复杂度**: O(L)，L=响应长度。

### 10. Agent评估指标
- **原理**: 计算标准分类指标 — TP/FP/TN/FN混淆矩阵 → 准确率(Accuracy)、精确率(Precision=TP/(TP+FP))、召回率(Recall=TP/(TP+FN))、F1=2×P×R/(P+R)；幻觉率=输出中无法从ground_truth溯源的声明占比。
- **时间复杂度**: O(N)，N=输出条目数。

## 使用示例

```python
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

# ReAct推理循环
tools = {"calculator": lambda x: str(eval(x))}
result = react_reasoning_loop("计算 (3+5)*2", tools, max_iterations=5)

# 链式思维求解
steps = [
    {"description": "解析括号"},
    {"description": "计算3+5=8"},
    {"description": "计算8*2=16"},
]
result = chain_of_thought_solver("计算(3+5)*2", steps)

# Few-shot模板
examples = [{"input": "1+1", "output": "2"}, {"input": "2+3", "output": "5"}]
result = few_shot_template_engine("math", examples, "3+4")

# 工作流编排
tasks = {
    "A": {"name": "Task A", "action": lambda p: "A done"},
    "B": {"name": "Task B", "action": lambda p: "B done"},
}
deps = [("A", "B")]
result = agent_workflow_orchestrator(tasks, deps, {})

# 记忆管理
memory_manager_short_long_term("add_long", "Python是解释型语言", {})
result = memory_manager_short_long_term("retrieve_long", "Python", {"top_k": 5})

# 提示词优化
optimized = prompt_optimizer("帮我写文章", ["clarity", "role", "format"])

# 评估指标
outputs = [{"answer": "A", "response_time": 1.0}, {"answer": "B", "response_time": 0.5}]
truth = [{"answer": "A"}, {"answer": "B"}]
metrics = agent_evaluation_metrics(outputs, truth)
print(f"F1: {metrics['f1_score']}")
```

## 开发与测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定测试类
pytest tests/test_main.py::TestReactReasoningLoop -v

# 运行测试并显示覆盖率（需要 pytest-cov）
pytest tests/ -v --cov=.
```

### 代码检查

```bash
flake8 main.py tests/
```

### 直接运行自测

```bash
python main.py
```

## 依赖

核心功能无外部依赖（仅使用Python标准库: json, re, collections, time, math, functools, hashlib）。

开发/测试依赖: pytest>=7.0, flake8>=6.0（见 requirements.txt）。

## CI/CD

本仓库通过 GitHub Actions 进行自动化测试。每次 push 和 PR 到 main 分支时，将在 Python 3.10、3.11、3.12 三个版本上运行 flake8 代码检查和 pytest 测试。

详见 [.github/workflows/ci.yml](.github/workflows/ci.yml)。
