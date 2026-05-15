# Lightweight Adaptive Inference Runtime

这是一个面向演示和实验的自适应推理运行时，用最少的依赖实现“先便宜模型、再质量评估、必要时升级大模型、并把失败经验写回本地”的闭环。

项目的目标不是把回答做得最复杂，而是把“路由、评估、升级、记忆、可观测性”这几个环节串起来，形成一个可以持续改进的最小可用系统。

## 项目目标

- 先用低成本模型处理简单任务，尽量节省推理成本。
- 对模型回答进行自动判定，发现质量不足时立即升级。
- 把失败案例和路由反馈保存到本地 JSON 文件，供后续决策参考。
- 使用语义复杂度、历史失败率、长度和关键词等多个因子做路由，而不是只靠硬规则。
- 保持实现足够轻量，默认只依赖 Python 标准库。

## 整体工作流

用户输入查询后，系统会按下面的顺序执行：

1. `Router` 读取查询内容，并结合本地失败记忆、语义复杂度和关键词特征决定先走 `small` 还是 `large`。
2. `ModelManager` 调用对应模型生成初稿。
3. 如果路由到了 `large`，系统直接返回结果，并记录这次路由反馈。
4. 如果先走 `small`，`Evaluator` 会检查答案是否不可信、过短或被判官模型判为 `BAD`。
5. 如果评估结果是 `GOOD`，直接返回小模型答案，并记录成功反馈。
6. 如果评估结果是 `BAD`，`EscalationManager` 会把原问题、小模型草稿和错误原因一起交给大模型修复。
7. 最后把失败案例写入 `data/failure_memory.json`，同时记录这次路由是否正确。

## 目录结构

```text
sekleton/
├── main.py
├── demo_layer3.py
├── test_semantic_routing.py
├── runtime/
│   ├── runtime.py
│   ├── router.py
│   ├── semantic_router.py
│   ├── evaluator.py
│   ├── escalation.py
│   ├── memory.py
│   └── trace.py
├── models/
│   ├── model_manager.py
│   ├── small_model.py
│   └── large_model.py
├── prompts/
│   └── judge_prompt.py
├── data/
│   └── failure_memory.json
└── README.md
```

## 模块逐个说明

### `main.py`

这是命令行入口。

它做的事情很直接：

- 解析 `--query` 参数。
- 如果传了单次查询，就调用 `run_once()` 执行一次完整推理，并打印最终答案和元信息。
- 如果没有传查询，就进入交互模式，持续接收用户输入，直到输入 `exit` 或 `quit`。
- 每次执行都会创建一个 `AdaptiveInferenceRuntime` 实例，真正的路由、评估和升级逻辑都在运行时对象里完成。

核心函数说明：

- `run_once(query)`：单次运行模式，适合命令行测试。
- `interactive_mode()`：循环交互模式，适合连续试验。
- `main()`：解析参数并选择运行模式。

### `runtime/runtime.py`

这是整个系统的总调度器，也是最重要的编排层。

`AdaptiveInferenceRuntime` 把路由、模型调用、质量评估、升级和记忆写入全部串起来，负责把各模块组合成一个可运行的闭环。

它的执行阶段可以理解为：

- `ROUTER`：决定先用哪个模型。
- `EXECUTION`：调用模型生成答案。
- `EVALUATOR`：检查小模型答案质量。
- `ESCALATION`：必要时把任务升级给大模型。
- `MEMORY`：记录失败和反馈。

主要逻辑：

- 初始化 `Trace`、`FailureMemory`、`ModelManager`、`Router`、`Evaluator` 和 `EscalationManager`。
- `run(query)` 是主入口，返回一个包含 `answer` 和 `meta` 的字典。
- 如果路由结果是 `large`，系统直接返回大模型答案。
- 如果先走 `small`，则经过 `Evaluator` 判定后决定是直接返回还是升级。
- 在每条路径上都会写入路由反馈，便于后续统计准确率。

返回值中的 `meta` 主要用于观察路径和状态，例如：

- `path`：本次请求走过的处理链路。
- `escalated`：是否发生了升级。
- `quality`：小模型答案是否通过评估。
- `task_type`：路由器识别出的任务类型。
- `failure_saved`：失败案例是否落盘。
- `routing_accuracy`：当前历史路由准确率。

### `runtime/router.py`

这是路由决策层，决定一个查询先交给小模型还是大模型。

它不是只看关键词，而是把多个因子加权后再下决定：

- 任务类型失败率：当前任务类型过去是否容易失败。
- 全局失败率：历史整体是否偏向失败。
- 语义复杂度：由 `SemanticRouter` 计算的连续分数。
- 长度和结构复杂度：问题是否过长、是否包含复杂标记词。

关键常量：

- `TASK_TYPE_KEYWORDS`：用于识别任务类型，例如 coding、architecture、optimization、analysis、workflow。
- `COMPLEXITY_MARKERS`：用于快速判断查询是否带有复杂意图。

主要方法：

- `route(query, memory_stats=None)`：输出路由结果，包含目标模型、原因、语义分数和决策分数。
- `extract_task_type(query)`：从文本中提取任务类型。
- `get_failure_rate(task_type)`：根据历史失败案例估算某类任务的失败率。

路由规则可以概括为：

- `decision_score >= 0.55` 时路由到 `large`。
- 否则路由到 `small`。

这意味着系统默认偏向成本更低的小模型，但会在高风险任务上提前升级。

### `runtime/semantic_router.py`

这是语义复杂度预测器，负责把文本查询映射成一个 0 到 1 之间的复杂度分数。

它的实现方式很轻量，不依赖向量数据库或外部嵌入模型，而是直接用本地文本特征和余弦相似度做近似判断。

核心思路：

- 准备两组种子句子，一组代表简单任务，一组代表复杂任务。
- 把查询和种子句子都转换成词频向量。
- 计算查询与简单种子、复杂种子的平均相似度。
- 根据“复杂相似度减去简单相似度”得到最终复杂度分数。

主要方法：

- `predict_complexity(query)`：返回复杂度分数，越高越偏复杂任务。
- `add_easy_seed(text)`：动态增加简单种子。
- `add_hard_seed(text)`：动态增加复杂种子。
- `debug_similarity(query)`：输出和种子的相似度分布，便于调试路由判断。

这个模块的价值在于：即使没有训练好的向量模型，也能做一个“比关键词更像语义判断”的轻量路由信号。

### `runtime/evaluator.py`

这是答案质量评估器，用于判断小模型的输出是否可以直接返回。

它的判定是分层的，不是只靠一次 LLM 打分：

1. 先检查模型输出里是否带有明显的不自信标记，例如 `[UNCERTAIN_MODEL_OUTPUT]`。
2. 再检查答案是否过短或空白。
3. 最后把查询和回答交给判官模型做 `GOOD` / `BAD` 判断。

主要方法：

- `evaluate(query, response)`：返回 `GOOD` 或 `BAD`。
- `_has_model_uncertainty_marker(response)`：识别模型自己暴露的不确定信号。
- `_is_too_short_or_empty(response)`：检查回答是否太短或为空。
- `_normalize_label(judge_output, response)`：把判官输出归一化成标准标签。

判官提示词内置在类变量 `JUDGE_PROMPT_TEMPLATE` 中，要求模型只输出 `GOOD` 或 `BAD`。

这套设计的核心目的是：让系统尽量在低成本阶段就识别出低质量答案，减少无谓的升级和错误传播。

### `runtime/escalation.py`

这是升级管理器，负责把失败的小模型回答交给大模型修复。

它不会简单地把原查询再发一遍，而是会把“为什么失败”一起带给大模型，帮助大模型做定向修复。

主要方法：

- `retry_with_large_model(query, small_answer, reasons)`：构造修复上下文，调用大模型生成改进版答案。

它会拼出一个修复提示，包含：

- 原始用户问题。
- 小模型的弱草稿。
- 评估器给出的质量问题原因列表。

这样做的好处是，大模型不是从零开始回答，而是针对“草稿中的缺陷”补全、修正和增强。

### `runtime/memory.py`

这是本地失败记忆和路由反馈存储层，所有历史数据都落在 `data/failure_memory.json`。

它的职责有两类：

- 记录失败案例，保存问题、小模型草稿、最终答案和失败原因。
- 记录路由反馈，统计路由到底是不是“选对了模型”。

主要方法：

- `load_memory()`：读取 JSON 文件并补齐缺失字段。
- `save_failure(...)`：保存一次失败样本。
- `get_failure_rate(...)`：按任务类型或模型估算失败率。
- `summarize_memory()`：汇总失败数量、按任务类型分布、按模型分布。
- `save_routing_feedback(...)`：记录一次路由决策是否正确。
- `get_routing_stats()`：计算路由准确率、最近趋势和按模型统计。

这里最重要的两个数据集是：

- `failures`：保存“答错了”的样本。
- `routing_feedback`：保存“路由选得对不对”。

这意味着系统既能记住答案失败，也能记住决策失败。

### `runtime/trace.py`

这是运行日志和可观测性模块。

它的作用不是影响决策，而是把每一步发生了什么打印出来，并保存到内存历史中，方便调试和展示。

主要方法：

- `log(stage, message, meta=None)`：打印带阶段标签、时间戳和 JSON 元数据的日志。
- `print_runtime_summary()`：把整个运行过程按阶段汇总出来。
- `clear_history()`：清空历史日志。
- `get_history()`：获取当前日志历史副本。

日志阶段包括：

- `ROUTER`
- `EXECUTION`
- `EVALUATOR`
- `ESCALATION`
- `MEMORY`
- `RUNTIME`

这让系统在演示时非常容易看懂每一步的决策链。

### `models/model_manager.py`

这是模型总管理器，统一封装小模型和大模型的调用入口。

它做了两件事：

- 初始化 `SmallModel` 和 `LargeModel`。
- 根据 `model_name` 分发到对应模型的 `generate()` 方法。

主要方法：

- `generate(model_name, query)`：对外统一调用接口。

如果模型调用失败，它会捕获异常并返回系统错误字符串，避免整个运行时直接崩掉。

### `models/small_model.py`

这是小模型封装，支持两种模式：

- 如果本地 Ollama 可用，就调用真实模型。
- 如果不可用，就退回到 Mock 逻辑。

它的重点不是“尽可能聪明”，而是“尽可能便宜、快、并能识别自己不确定”。

主要机制：

- 启动时先测试 Ollama 服务是否可用。
- 如果服务可用，调用本地 `http://localhost:11434/api/chat` 接口。
- 如果模型输出表现出不自信，会在结果前加上 `[UNCERTAIN_MODEL_OUTPUT]` 标记。
- 如果长问题的回答过短，会在结果前加上 `[SHORT_RESPONSE]` 标记。
- 如果 Ollama 不可用，则进入 Mock 生成逻辑，用关键词和长度规则模拟一个低成本模型。

主要方法：

- `generate(query)`：统一入口。
- `_test_connection()`：检测 Ollama 服务是否在线。
- `_call_ollama(query, temperature, max_tokens)`：发送原生 HTTP 请求。
- `_generate_with_ollama(query)`：真实推理路径。
- `_generate_with_mock(query)`：离线降级路径。
- `_detect_uncertainty(response)`：检查回答里有没有不自信表述。
- `_generate_low_confidence_response(reason)`：返回低置信度草稿。
- `_generate_high_confidence_response()`：返回一个更像“简单任务成功回答”的固定文本。

当前 `ModelManager` 初始化时把 `fallback_to_mock` 设为 `False`，但由于小模型在连接失败时会自动转入本地 Mock 路径，所以项目仍然能在没有 Ollama 的情况下跑起来，只是结果会更像演示文本。

### `models/large_model.py`

这是大模型封装，负责在需要升级时生成更高质量的答案。

它同样通过本地 Ollama 的 HTTP 接口调用模型，但会根据输入内容区分两类场景：

- 常规高难度请求：直接作为技术问答处理。
- 修复请求：当小模型失败后，带着弱草稿和质量问题一起修复。

主要方法：

- `generate(query)`：判断是常规请求还是修复请求。
- `_handle_regular(query)`：处理普通高难度问题。
- `_handle_repair(repair_context)`：处理升级修复请求。
- `_call_ollama(messages, temperature)`：发送本地模型请求。

它和小模型的区别是：

- 系统更偏向用它做最终修复。
- 提示词更强调“结构化、具体、完整”。
- 在修复模式下会显式要求解决弱草稿的问题，不要道歉，只输出改进后的答案。

如果本地 Ollama 不可用，大模型路径会返回系统错误信息，因此如果你希望完整体验升级链路，最好启动本地 Ollama 服务。

### `prompts/judge_prompt.py`

这是一个独立的判官提示词常量。

虽然 `Evaluator` 里也有自己的完整模板，但这个文件保留了一个更简洁的质量判断提示，用于表达“质量评估只输出 GOOD 或 BAD”这一意图。

主要内容：

- 要求判官按相关性、完整性、自信度、可执行价值来分类。
- 明确限制输出只能是 `GOOD` 或 `BAD`。

### `data/failure_memory.json`

这是本地持久化数据文件。

它保存两类信息：

- `failures`：失败样本，包括问题预览、任务类型、选择了哪个模型、失败原因、小模型草稿和最终答案。
- `routing_feedback`：路由反馈，包括预测复杂度、选择模型、最终质量、是否选对。

此外还有 `metadata`，用于记录数据版本。

这个文件的存在让系统具备“越跑越有历史”的能力，也让你可以在离线环境下分析路由行为。

### `demo_layer3.py`

这是演示脚本，用来展示 Layer 3 的完整路由与学习能力。

它主要做四件事：

- 演示语义复杂度分析。
- 演示多因子加权路由。
- 演示路由反馈记录和统计。
- 解释整个自学习闭环的架构。

主要函数：

- `demo_semantic_complexity()`：展示简单任务和复杂任务的分数差异。
- `demo_multifactor_routing()`：展示路由器如何根据多个因子做决定。
- `demo_routing_feedback()`：模拟写入路由反馈并打印统计结果。
- `demo_layer3_learning()`：输出 Layer 3 学习系统的架构说明。

如果你想快速理解这个项目如何“越用越聪明”，先跑这个脚本最合适。

### `test_semantic_routing.py`

这是集成测试脚本，用来验证语义路由和反馈记录是否正常工作。

它的验证重点是：

- 简单和复杂问题是否会走不同路径。
- 路由反馈是否成功写入 JSON 文件。
- 失败记忆和路由统计是否能正常聚合。

主要函数：

- `print_section(title)`：打印测试分段标题。
- `test_semantic_routing()`：执行完整集成测试。

这个脚本特别适合在你修改路由、记忆或运行时编排后做一次快速回归检查。

### `runtime/__init__.py`、`models/__init__.py`、`prompts/__init__.py`

这三个文件当前都是空的。

它们的作用只是把对应目录标记成 Python 包，方便模块导入，比如：

- `runtime.runtime`
- `models.model_manager`
- `prompts.judge_prompt`

## 运行方式

### 1. 单次查询

```bash
python main.py --query "Design an adaptive runtime for LLM inference"
```

适合快速测试一条输入从路由到返回的完整链路。

### 2. 交互模式

```bash
python main.py
```

进入后可以持续输入问题，输入 `exit` 或 `quit` 退出。

### 3. 演示脚本

```bash
python demo_layer3.py
```

这个脚本会展示语义路由、加权决策、反馈记录和学习闭环。

### 4. 集成测试

```bash
python test_semantic_routing.py
```

用于验证路由、升级和统计是否正常。

## 运行时输出示例

```text
[RUNTIME] 10:31:02.123 | received query | {"query": "Design adaptive runtime..."}
[ROUTER] 10:31:02.124 | selected_model=small | {"task_type": "architecture", "decision_score": 0.432, "semantic_score": 0.365}
[EXECUTION] 10:31:02.501 | invoking model=small | {"route_reason": "default_cost_first"}
[EVALUATOR] 10:31:03.020 | BAD
[ESCALATION] 10:31:03.021 | escalating to large model with repair context | {"query_len": 48, "reasons": ["judge_bad"]}
[MEMORY] 10:31:03.812 | failure case saved | {"task_type": "architecture", "reason": ["judge_bad"]}
```

这些日志来自 `Trace`，可以帮助你看清每一步为什么这么走。

## 数据结构说明

### `failures` 样本

`save_failure()` 会写入如下字段：

- `timestamp`：时间戳。
- `task_type`：任务类型。
- `selected_model`：当时选中的模型。
- `query_preview`：查询预览。
- `reason`：失败原因列表。
- `small_answer`：小模型草稿，若有。
- `final_answer`：最终答案，若有。

### `routing_feedback` 样本

`save_routing_feedback()` 会写入如下字段：

- `timestamp`：时间戳。
- `query_preview`：查询预览。
- `predicted_complexity`：路由器预测复杂度。
- `selected_model`：最终选中的模型。
- `final_quality`：`GOOD` 或 `BAD`。
- `was_correct`：这次路由是否正确。

## 设计特点

- 轻量：只依赖 Python 标准库。
- 本地：数据全部落到本地 JSON 文件。
- 可观测：每个阶段都有 trace 日志。
- 可学习：失败样本和路由反馈都会累积。
- 可替换：如果以后要接入真实 API 或更强的模型，只需要替换 `models/` 里的实现。

## 当前限制

- 小模型和大模型都默认走本地 Ollama 接口，需要本机服务可用才能体验真实推理。
- 如果没有 Ollama，大模型路径会返回系统错误字符串。
- 语义路由仍是轻量近似实现，不是训练好的嵌入模型。
- 当前学习机制是统计和记录型的，还没有自动调阈值的在线优化逻辑。

## 后续可扩展方向

- 给路由器增加动态阈值调整。
- 根据历史失败样本自动更新复杂种子。
- 为大模型修复阶段加入更细粒度的错误分类。
- 把本地 JSON 存储替换成更稳健的数据库或向量存储。

## 一句话总结

这个项目实现了一个最小但完整的自适应推理闭环：先路由、再生成、再评估、必要时升级、最后把经验写回记忆，形成一个可以持续迭代的本地智能体运行时。
