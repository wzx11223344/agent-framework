---
name: agent-framework-zx
displayName: AI Agent框架
summary: 10个Agent算法：ReAct推理循环/链式思维回溯/Few-shot模板引擎/DAG工作流编排/工具注册分派/LRU+TF-IDF记忆管理/提示词优化/多Agent协调/响应解析修复/评估指标
tags:
  - ai
  - agent
  - llm
  - reasoning
  - workflow
---

# agent-framework-zx

AI Agent 框架工具集，提供10个Agent开发核心算法（纯标准库实现）。

## 功能概述

1. **react_reasoning_loop** - ReAct推理循环（Thought→Action→Observation + 推理链追踪）
2. **chain_of_thought_solver** - 链式思维求解器（分步推理 + 中间验证 + 回溯）
3. **few_shot_template_engine** - Few-shot模板引擎（Jaccard相似度选择 + 模板填充）
4. **agent_workflow_orchestrator** - 工作流编排器（DAG + Kahn拓扑排序 + 并行/串行/条件分支）
5. **tool_registry_and_dispatcher** - 工具注册分派（签名验证 + MD5缓存 + 错误处理）
6. **memory_manager_short_long_term** - 记忆管理（LRU短期 + TF-IDF长期检索）
7. **prompt_optimizer** - 提示词优化器（结构分析 + 模糊性/上下文/角色检查）
8. **multi_agent_coordinator** - 多Agent协调器（协作/竞争/层级协议）
9. **response_parser_and_validator** - 响应解析验证器（JSON/Markdown解析 + 自动修复）
10. **agent_evaluation_metrics** - Agent评估指标（准确率/精确率/召回率/F1/幻觉率）

## 依赖

无外部依赖（仅使用Python标准库: json, re, collections, time, math, functools, hashlib）。
