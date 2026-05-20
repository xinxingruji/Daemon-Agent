# basic_skeleton — Agent 框架 总览

这是一个本地化、轻量但完整的“自适应推理 + Agent”框架，实现了从路由、模型调用、评估到升级及失败记忆的闭环。

下面文档在一个文件里描述：组成、运行流程、模块说明、每个重要文件用途，以及与 `learn-claude-code/agents/s_full.py` 的对比。

## 快速开始

1. 推荐先准备本地 Ollama 与模型（可选，但能得到真实推理质量）：

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen3.5:9b
ollama start
```

2. 启动 Agent（完整模式，已集成自适应推理）：

```bash
python main.py --mode agent
```

3. 调试/单次推理：

```bash
python main.py --mode inference --query "Explain adaptive runtime"
python main.py --mode inference    # 交互式
```

环境变量（可选）：

- `OLLAMA_BASE_URL`：默认 `http://localhost:11434/api/chat`
- `AGENT_MODEL` / `COMPRESSION_MODEL`：覆盖默认模型 id

## 项目总体架构（高层）

- 入口：`main.py`（支持 `agent` 与 `inference` 两个模式）
- Agent 层：`agent.py`（多轮循环、工具分派、上下文管理、与 runtime 集成）
- 推理运行时：`runtime/runtime.py`（AdaptiveInferenceRuntime，负责路由→执行→评估→升级→记忆）
- 模型封装：`models/`（`small_model.py`、`large_model.py`、`model_manager.py`）
- 工具与协作：`TodoManager`、`BackgroundManager`、`MessageBus`、`TeammateManager`（在 `agent.py` 中暴露工具）
- 持久化：`data/failure_memory.json`（失败样本与路由反馈）
- 帮助与示例：`demo_layer3.py`、`test_semantic_routing.py`、`test_integrated_agent.py`

## 运行流程（逐步详细）
1. 接收输入

- 用户在 REPL 输入一条查询，或外部系统调用 `run_once(query)`。系统把该查询追加到当前会话的 `messages` 历史中（`role: user`）。

2. 预处理（Agent 层）

- 执行 `microcompact(messages)`：清除临时/重复信息，保持短期上下文精简。
- 调用 `BG.drain()` 拉取后台任务通知并把它们作为系统消息注入对话。
- 检查收件箱 `BUS.read_inbox('lead')` 并把未读消息作为系统/assistant 消息注入。
- 如果存在未完成的 `TODO` 项，Agent 根据策略决定是否在每轮末尾触发提醒。

3. 自适应推理（Runtime 层）

- Agent 调用 `AdaptiveInferenceRuntime.run(query)`（或把其作为工具在对话中显式调用）。运行时内部执行：
	- Router：调用 `semantic_router.predict_complexity()` + 查询历史失败率 + 长度/关键词特征，计算 `decision_score` 和 `semantic_score`，返回路由决策（`target`）。
	- Execution：若 `target == 'small'` 调用 `ModelManager.generate('small', query)`；若 `target == 'large'` 则直接调用大模型。
	- Evaluator：对小模型输出执行 `Evaluator.evaluate(query, small_answer)`，返回 `GOOD` / `BAD`。
	- Escalation：若评估为 `BAD`，调用 `Escalation.retry_with_large_model(query, small_answer, reasons)`，并把大模型输出作为最终候选。
	- Memory：把失败样本与路由反馈写入 `data/failure_memory.json`，并更新路由统计。

4. 注入与模型主循环（Agent 层）

- Runtime 返回 `{'answer': str, 'meta': {...}}`（`meta` 含 `path`、`escalated`、`quality`、`routing_score` 等）。Agent 将 `answer` 或 `meta` 以系统消息形式注入 `messages`，并发起下一次模型调用（若需要）。

5. 工具调用执行

- 如果模型内容包含 `tool_use`（例如模型直接返回一段 JSON 指示执行 `read_file`），Agent 会：
	1) 使用 `extract_json_block()` 从回复中解析工具调用结构：`{'name':'read_file','input':{'path':'data/x.csv'}}`。
	2) 调用 `TOOL_HANDLERS['read_file'](**input)`，将返回结果作为 `role: tool` 的消息追加到 `messages`。
	3) 将工具结果再次交给模型（继续对话），模型据此生成后续动作或最终回答。

6. 结束/压缩与记录

- 本轮结束前，Agent 可能触发 `auto_compact(messages)` 将长历史压缩为摘要并把完整 transcript 保存到 `.transcripts/`。
- 如果发生失败或升级，运行时会把 `failure` 和 `routing_feedback` 写入 `data/failure_memory.json`，便于后续统计与动态调整。

示例演示（完整调用链）

场景：用户请求 "请读取 data/sales.csv，计算 price 列的平均值并保存到 results.txt"。这个请求既包含文件 I/O 又包含计算，可能导致先用小模型尝试，再升级到大模型。

1) 用户输入：

```
User: 请读取 data/sales.csv，计算 price 列的平均值并保存到 results.txt
```

2) Router（示例输出）：

```json
{
	"semantic_score": 0.32,
	"decision_score": 0.28,
	"target": "small",
	"reasons": ["成本优先","非复杂算法"]
}
```

3) Small model 生成（示例回复，含工具调用）：

```json
{"type":"tool_use","name":"read_file","input":{"path":"data/sales.csv","limit":500}}
```

Agent 执行 `TOOL_HANDLERS['read_file'](path='data/sales.csv', limit=500)`，返回 CSV 内容（文本），并把该文本作为 `role: tool` 消息追加到 `messages`。

4) Small model 基于文件内容生成草稿回答，但检测到不自信（示例标记）：

```
[UNCERTAIN_MODEL_OUTPUT] 我读取了文件，但对数据格式有些不确定。我建议先尝试下面的 Python 代码来计算平均值：
```python
import csv
...
```
```

5) Evaluator 判定为 `BAD`（因包含 `[UNCERTAIN_MODEL_OUTPUT]`）：

```json
{"result":"BAD","reasons":["uncertain_marker","short_answer"]}
```

6) Escalation 发起到大模型：把 `query`、`small_answer` 与 `reasons` 构造成 `repair_context`，调用 `LargeModel.generate(repair_context)`，大模型返回完整、可靠的代码与 `results.txt` 内容。

7) Agent 执行 `TOOL_HANDLERS['write_file'](path='results.txt', content='average: 123.45')`，并把结果记录为工具消息。

8) Runtime 将一次失败样本写入 `data/failure_memory.json`：

```json
{
	"timestamp": 1716090000,
	"task_type": "file_compute",
	"selected_model": "small",
	"reason": ["uncertain_marker"],
	"small_answer": "[UNCERTAIN_MODEL_OUTPUT] ...",
	"final_answer": "计算并写入 results.txt: average: 123.45"
}
```

9) 最终 Agent 返回给用户：成功确认与结果位置，并在 `meta` 中指出本次请求走过的路径：

```json
{
	"answer": "已计算并把平均值写入 results.txt（average: 123.45）",
	"meta": {"path":"router->small->evaluator->escalation->large->tool_write","escalated":true}
}
```

说明：上述示例展示了路由优先成本的策略（先用小模型）、小模型自检导致升级、大模型修复输出、工具执行以及失败记忆的写入完整闭环。

## 关键模块说明（文件级别）

- `main.py`：命令行入口，解析 `--mode` 与 `--query`；`agent` 模式启动多轮 REPL；`inference` 模式用于单次或交互性调试 `AdaptiveInferenceRuntime`。

- `agent.py`：实现完整 Agent loop（microcompact、工具分派、TODO 管理、后台任务等），并在每轮中集成 `AdaptiveInferenceRuntime` 的调用。暴露 `TOOL_HANDLERS` 与 `TOOLS` schema，支持文件读写、bash、task、TodoWrite、inference 等。

- `runtime/runtime.py`：`AdaptiveInferenceRuntime`。主入口 `run(query)` 返回 `{'answer': str, 'meta': {...}}`。协调 `router`、`model_manager`、`evaluator`、`escalation`、`memory` 与 `trace`。

- `runtime/router.py`：多因子路由实现，输出目标模型、语义分数与决策分数；阈值化决定 small/large。

- `runtime/semantic_router.py`：轻量语义复杂度预测（TF-like 向量 + 余弦与种子句子比较），返回 0-1 分数。

- `runtime/evaluator.py`：答案质量评估器（自信度标记、长度检查、判官模型 GOOD/BAD）。

- `runtime/escalation.py`：升级修复逻辑，把小模型的弱草稿与失败原因提交给大模型以获得修复答案。

- `runtime/memory.py`：`failure_memory.json` 的读写接口，保存 `failures` 与 `routing_feedback`，并提供统计方法 `get_routing_stats()`。

- `runtime/trace.py`：可观测性与日志记录，按阶段记录事件并支持汇总展示。

- `models/small_model.py`：小模型封装，优先调用本地 Ollama（若可用），否则降级到 Mock；实现自信度检测并在必要时添加标记（例如 `[UNCERTAIN_MODEL_OUTPUT]`）。

- `models/large_model.py`：大模型封装，负责常规高难度请求与修复请求；修复模式下接收 `repair_context` 并返回改进答案。

- `models/model_manager.py`：统一模型调用接口，封装 small/large 的生成逻辑并捕获异常。

- `prompts/judge_prompt.py`：判官提示模板，要求输出 `GOOD` 或 `BAD`。

- `demo_layer3.py`：演示脚本，说明路由、反馈与学习闭环的运行。

- `test_semantic_routing.py`、`test_integrated_agent.py`：用于单元/集成测试，验证路由、反馈与 agent 流的正确性。

## 每个文件详解（详细）

> 下面以文件为单位逐项说明实现细节、重要函数签名、输入/输出格式示例和扩展点。

- `main.py`
	- 主要函数：
		- `parse_args()`：处理 `--mode`、`--query`、`--verbose` 等 CLI 参数。
		- `run_once(query)`：创建 `AdaptiveInferenceRuntime` 并返回运行结果（`{'answer': str, 'meta': {...}}`）。
		- `interactive_mode()`：REPL，按行读取用户输入并调用 `run_once` 或启动 Agent loop。
	- 输出/示例：
		```python
		result = run_once('How to design an adaptive runtime?')
		print(result['answer'])
		print(json.dumps(result['meta'], indent=2))
		```
	- 扩展点：可在外部脚本中直接导入 `run_once` 用作批量测试。

- `agent.py`
	- 关键结构：`Agent` 类、`TOOL_HANDLERS` 字典、`TOOLS` schema 列表、`TodoManager`、`BackgroundManager`。 
	- 主要流程（伪码）：
		```text
		while True:
			microcompact(messages)
			runtime_result = AdaptiveInferenceRuntime.run(query)
			messages.append({'role':'system','content': runtime_result['answer']})
			resp = call_model(messages)
			if resp包含 tool_use:
				 for tool_call in resp.tool_calls:
						 out = TOOL_HANDLERS[tool_call['name']](**tool_call['input'])
						 messages.append({'role':'tool','content': out})
			else:
				 messages.append({'role':'assistant','content': resp_text})
		```
	- 工具接口约定：每个 handler 接受关键字参数并返回字符串（或可 JSON 序列化的对象），例如：
		```python
		def read_file(path: str, limit: int=None) -> str: ...
		TOOL_HANDLERS['read_file'] = lambda **kw: read_file(kw['path'], kw.get('limit'))
		```
	- 注意：Models 的 `tool_use` 输出需保持稳定的 JSON 格式；Agent 有辅助函数 `extract_json_block(text)` 用于从模型回复中提取工具调用参数。

- `runtime/runtime.py`（`AdaptiveInferenceRuntime`）
	- 主要方法：
		- `run(query: str) -> dict`：主入口，返回 `{'answer': str, 'meta': {path, escalated, quality, routing_accuracy, ...}}`。
		- 内部使用 `Router.route(query)` 返回 `{'target': 'small'|'large', 'score': float, 'reason': str}`。
	- 路由到小模型时会走 `Evaluator.evaluate(query, small_answer)`，返回 `GOOD`/`BAD`；若 `BAD` 则调用 `Escalation.retry_with_large_model(...)`。
	- `meta` 示例：
		```json
		{"path": "router->small->evaluator->escalation->large",
		 "escalated": true,
		 "quality": "GOOD",
		 "routing_accuracy": 0.72}
		```

- `runtime/router.py`
	- 输入：`route(query: str, memory_stats: dict=None)`；输出含 `target`, `decision_score`, `semantic_score`, `reasons`。
	- 可定制权重：语义复杂度、失败率、长度、关键词等权重可在模块顶部配置。

- `runtime/semantic_router.py`
	- 提供 `predict_complexity(query: str) -> float`（0-1），以及维护 `easy_seeds` / `hard_seeds` 的接口。
	- 实现细节：TF-like 词频向量、L2 归一化、余弦相似度。

- `runtime/evaluator.py`
	- `evaluate(query: str, response: str) -> 'GOOD' | 'BAD'`。
	- 判官调用示例（内部）：
		```python
		judge_out = call_judge_model(prompt=JUDGE_PROMPT_TEMPLATE.format(query=query, answer=response))
		return normalize(judge_out)
		```

- `runtime/escalation.py`
	- `retry_with_large_model(query, small_answer, reasons) -> str`：构造修复提示并调用大模型。

- `runtime/memory.py`
	- 接口：`save_failure(failure_obj)`, `save_routing_feedback(feedback_obj)`, `get_routing_stats()`。
	- `failure_obj` 示例：
		```json
		{"timestamp":123, "task_type":"coding", "selected_model":"small",
		 "reason":["judge_bad"], "small_answer":"...", "final_answer":"..."}
		```

- `models/small_model.py` / `models/large_model.py`
	- 共有方法：`generate(query: str, **opts) -> str`。
	- Ollama 请求样例（HTTP body）：
		```json
		{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"..."}],"temperature":0.2}
		```
	- 返回解析：解析 `choices` / `content` 字段，将文本返回；若包含结构化 `tool_use` 则保留原文以供 `Agent` 解析。

- `models/model_manager.py`
	- `generate(model_name: str, query: str) -> str`，内部调用对应模型实例并捕获异常。
	- 可扩展：增加 `ModelBackend` 抽象并在此注册。

- `prompts/judge_prompt.py`
	- 提供 `JUDGE_PROMPT_TEMPLATE` 字符串，要求判官输出 `GOOD` 或 `BAD`（便于 `Evaluator` 自动解析）。

## 调试与常见问题（快速诊断）

- 检查 Ollama 是否运行：

```bash
curl -sS $OLLAMA_BASE_URL || echo "Ollama not reachable"
```

- 检查本地模型是否拉取：

```bash
ollama ls  # 列出已安装模型
```

- 运行集成测试查看模块健康：

```bash
python test_integrated_agent.py
```

- 如果 `WinError 10061`，说明本地 Ollama 未运行或端口被占用；启动 Ollama 或调整 `OLLAMA_BASE_URL`。

## 扩展指南（快速）

- 添加新工具：在 `agent.py` 中实现处理函数并注册到 `TOOL_HANDLERS`，同时在 `TOOLS` 中添加 schema。

- 添加新模型后端：实现 `ModelBackend`（提供 `generate(query)`）并在 `model_manager.py` 中注册映射（`'mybackend' -> MyBackend()`）。

- 支持其他后端（Anthropic/OpenAI）：在 `models/` 下新增适配器模块并在 `model_manager` 中按配置选择后端。

## 数据与持久化

## 数据与持久化

- `data/failure_memory.json`：包含 `failures`（失败样本）和 `routing_feedback`（路由反馈）以及 `metadata`。用于统计与后续策略调整。

## 日志与可观测性

- `trace` 输出示例（在控制台或日志文件）：

```
[RUNTIME] ... received query
[ROUTER] ... selected_model=small
[EVALUATOR] ... BAD
[ESCALATION] ... escalating to large
[MEMORY] ... failure saved
```

这些信息帮助理解每条请求为何走某条路径。

## 如何在没有 Ollama 时运行（降级策略）

- `SmallModel` 支持自动降级到 mock 模式（关键词+模板），以便在没有本地模型时仍能演示和测试。
- `LargeModel` 在没有 Ollama 时会返回错误信息，因此建议在需要完整升级链路时启动 Ollama。

## 测试命令

```bash
python test_semantic_routing.py
python test_integrated_agent.py
python demo_layer3.py
```

## 与 `learn-claude-code/agents/s_full.py` 的对比（简明）

相同点：
- 都实现了完整 agent loop（预处理、LLM 调用、工具分派、子 agent 支持、Todo 管理、上下文压缩）。
- 都提供工具集合（文件 I/O、bash、task、TodoWrite 等）。
- 都保存对话 transcript 并支持自动压缩与长期历史。

关键差异：
- LLM 接口：`s_full.py` 以 Anthropic 客户端为中心，直接把 `TOOLS` schema 交给模型，由模型发起 `tool_use` 事件流；本仓库优先使用本地 Ollama HTTP 接口並在 agent 端显式解析模型输出與触发工具处理（更显式的 handler 控制）。
- 推理分层：本项目把路由/评估/升级抽象为独立的 `AdaptiveInferenceRuntime`（模块化、可复用），`s_full.py` 更倾向把控制流嵌入主循环並依赖模型 prompt 控制策略。
- 多模型策略：本项目明确区分压缩模型与主模型（`COMPRESSION_MODEL` / `AGENT_MODEL`），在 runtime 层做分层路由；`s_full.py` 通常以单一 MODEL 常量为主并通过提示词管理行为。
- 错误与降级：本仓库实现了 SmallModel 的自动降级 mock 路径與 FailureMemory 的统计回路；`s_full.py` 示例更多演示同步工具流與团队协作功能。
- 可测试性：保留独立的 `inference` 入口用于调试 `AdaptiveInferenceRuntime`，便于在不启动完整 agent loop 的情况下测试路由与升级逻辑。

建议：若你需要与 `s_full.py` 更高保真地兼容，可考虑在 `agent` 中添加对 Anthropic SDK 的适配层（Adapter），或让 `AdaptiveInferenceRuntime` 支持多后端（Ollama/Anthropic/OpenAI）。

---

已把项目的详细说明合并到本 `README.md` 中；如果你希望我把这份文档拆成 `AGENT_GUIDE.md`（更长）並把 README 保留为简短入口，我也可以按你的偏好拆分並生成目录索引。

