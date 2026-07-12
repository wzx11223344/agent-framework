# agent-framework

AI Agent 框架工具集 - 提供10个Agent开发常用功能，无外部依赖。

## 功能列表

| # | 函数 | 说明 |
|---|------|------|
| 1 | `prompt_template_manager` | 提示词模板管理（增删改查） |
| 2 | `chain_of_thought_solver` | 链式思维求解器 |
| 3 | `few_shot_classifier` | Few-shot分类器 |
| 4 | `agent_workflow_builder` | Agent工作流构建（DAG拓扑排序） |
| 5 | `response_parser` | LLM响应解析器（JSON/Markdown/XML） |
| 6 | `token_counter` | Token计数器（多模型支持） |
| 7 | `conversation_summarizer` | 对话摘要生成 |
| 8 | `prompt_optimizer` | 提示词优化器 |
| 9 | `multi_agent_coordinator` | 多Agent协调器 |
| 10 | `eval_metric_calculator` | 评估指标计算（accuracy/precision/recall/f1/mse/mae） |

## 安装依赖

本项目仅使用Python标准库，无需安装外部依赖。

## 使用示例

```python
from main import (
    prompt_template_manager,
    chain_of_thought_solver,
    agent_workflow_builder,
    response_parser,
    token_counter,
    eval_metric_calculator,
)

# 模板管理
prompt_template_manager("create", "greeting", "Hello, {name}!", {"name": ""})
result = prompt_template_manager("read", "greeting", variables={"name": "World"})

# 链式思维
solution = chain_of_thought_solver("如何提高代码质量？")

# 工作流构建
workflow = agent_workflow_builder(
    steps=[
        {"id": "s1", "name": "数据收集", "depends_on": []},
        {"id": "s2", "name": "数据处理", "depends_on": ["s1"]},
        {"id": "s3", "name": "结果输出", "depends_on": ["s2"]},
    ],
    inputs={"data_source": "database"},
)

# Token计数
tokens = token_counter("这是一段测试文本", model="gpt-4")

# 评估指标
metrics = eval_metric_calculator(
    predictions=[1, 0, 1, 1, 0],
    ground_truth=[1, 0, 0, 1, 1],
    metric="f1",
)
```

## 依赖

无外部依赖（仅使用Python标准库）。
