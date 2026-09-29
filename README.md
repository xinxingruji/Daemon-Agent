# 🚀 如何运行 (Usage)

本系统依赖 **Ollama**（提供本地向量嵌入）和 **LiteLLM Proxy**（统一大小模型 API 调用网关）来运行。

---

## 1. 环境准备与依赖安装

项目支持 Python 3.11 及以上版本。克隆项目后，首先安装所需的 Python 依赖包：

```bash
pip install -r requirements.txt

```

---

## 2. 启动本地向量大脑 (Ollama)

我们需要启动 Ollama 服务，并拉取用于计算语义相似度的轻量级嵌入模型：

```bash
# 启动 Ollama (如果在后台运行则跳过)
ollama serve

# 拉取 nomic-embed-text 模型
ollama run nomic-embed-text-v2-moe

```

---

## 3. 配置并启动 LiteLLM 模型网关

我们的代码中动态路由了 `small` 和 `large` 两个模型名称。需要通过 LiteLLM 将它们映射到真实的模型（比如本地的 Qwen 和线上的 Claude）。

在项目根目录创建一个 `litellm_config.yaml` 文件，示例如下，仅供参考，请你改成自己的

```yaml
model_list:
  - model_name: small
    litellm_params:
      model: ollama/qwen3.5:9b
      api_base: http://localhost:11434
      extra_body:
        think: false
  - model_name: large
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: (your_api_key)

```

也可以直接复制模板：

```bash
cp litellm_config.example.yaml litellm_config.yaml

```


> [!IMPORTANT]
> **注意 1：** 这份 `litellm_config.yaml` 中绝对不能出现中文，连注释也不可以。
> 
> **注意 2：** 云端的 `api_key` 一定要写在这里，写在 `.env` 中没用，因为 litellm 不会去 `.env` 中读取。

> [!NOTE]
> **注意 3：** 本地小模型实测过于鸡肋了，可以配置 `small` 为便宜的云端模型，`large` 为贵的云端模型。
> 
> **注意 4：** 注意如果云端模型时 deepseek，`model:` 那里要写 `deepseek/deepseek-chat`（非思考）或者 `deepseek/deepseek-reasoner`（思考版）。具体详见: >https://github.com/BerriAI/litellm#supported-providers-website-supported-models--docs
> 
> **注意 5：** 如果 `small` 决定用本地小模型，注意在 `small` 部分的 `litellm_params` 下面加一个：
> ```yaml
> extra_body:
>         think: false
> 
> ```
> 
> 
> 这样可以关闭本地模型的思考模式。

配置好后**在一个新终端窗口启动 LiteLLM 代理**：

```bash
litellm --config litellm_config.yaml --port 4000

```

---

## 4. 预处理种子向量（首次运行前执行一次）

运行预处理脚本，将 `utterances.py` 中的种子文本一次性转为向量缓存：

```bash
python precompute_seeds.py

```

> 本脚本使用 8 线程并发调用 Ollama，约 10 秒完成当前 81 条种子。生成的 `seed_vectors.json` 会记录 schema 版本、embedding 模型、向量维度和内置种子哈希；Router 启动时会校验这些元数据。若模型或 `utterances.py` 发生变化，系统会明确提示重新运行本脚本。只有全部种子成功生成后才会替换原缓存。

---

## 5. 环境配置与启动主程序

### 环境变量配置

先复制一份环境配置模板：

```bash
cp .env.example .env

```

然后打开 `.env` 文件，进行如下配置。配置 Anthropic SDK 指向刚才启动的 LiteLLM 代理：

```env
ANTHROPIC_BASE_URL="http://localhost:4000"
ANTHROPIC_API_KEY="sk-placeholder"

# 可选，以下为默认值
EMBEDDING_MODEL="nomic-embed-text-v2-moe"
OLLAMA_EMBEDDING_URL="http://localhost:11434/api/embeddings"
EMBEDDING_CACHE_SIZE="256"

```

### 运行前诊断

启动 Agent 前可执行只读健康检查：

```bash
python doctor.py
```

该命令检查 Python 与依赖、`.env`、small/large 模型映射、种子缓存、Ollama embedding 模型和 LiteLLM 端口，不会输出 API Key。若只想检查本地文件和配置，不访问本地服务：

```bash
python doctor.py --offline
```

### 运行主程序

**一切就绪！现在，运行主程序：**

```bash
python main.py

```

进入 `Daemon >>` 终端后，输入你的任务。支持 `!large` / `!small` 前缀强制指定模型，`/reload` 热重载种子库。

命令工具默认以参数数组执行，不启用隐式 shell。管道、重定向、删除、网络访问、依赖安装和 Git 远端操作等高风险命令会返回一次性审批编号，不会立即执行。可使用：

```text
/approvals
/approve <request-id>
/deny <request-id>
```

审批只能由终端用户完成，模型不能批准自己的命令。Agent 文件工具也会阻止直接访问 `.env`、真实 `litellm_config.yaml` 和 `.git`；请通过人工编辑或受控 Git 命令管理这些文件。

---

## 6. 测试

本项目提供两类测试：正确性测试（pytest，纯逻辑验证）和性能基准测试（benchmark，含本地 + API 两种模式）。

### 6.1 正确性测试（不依赖外部服务）

```bash
python -m pytest tests/ -v
```

共收集 91 项测试。默认运行其中 89 项离线测试，并跳过 2 项真实模型 API 延迟测试；添加 `--run-api` 后运行全部 91 项。离线测试不依赖 Ollama、不调用模型 API、不写入项目运行数据，通过依赖注入、mock、子进程和临时目录隔离外部依赖。

**测试内容：**

| 分类 | 文件 | 测试数 | 验证点 |
|------|------|--------|--------|
| 种子数据完整性 | [tests/test_utterances.py](tests/test_utterances.py) | 4 | SMALL/LARGE 非空、无重复 |
| 余弦相似度 | [tests/test_router.py](tests/test_router.py) | 4 | 相同向量 → 1.0、正交 → 0、相反 → -1.0、空向量 → 0 |
| 路由决策 | [tests/test_router.py](tests/test_router.py) | 6 | force_large/force_small 强制升级/降级、空 query 保守兜底、错题本拦截、低于阈值降级、token 惩罚动态升阶 |
| 错题本管理 | [tests/test_router.py](tests/test_router.py) | 2 | 失败记录写入 JSONL、达到容量时触发后台压缩 |
| 种子库管理 | [tests/test_router.py](tests/test_router.py) | 5 | 添加种子、去重、空文本跳过、按相似度删除、热重载 |
| 性能基准 | [tests/test_benchmarks.py](tests/test_benchmarks.py) | 13 | 初始化耗时、余弦速度、路由延迟、错题本规模影响、路由准确率、成本模拟、API 延迟多次测量 |
| 缓存 schema | [tests/test_seed_cache.py](tests/test_seed_cache.py) | 6 | 元数据、旧格式兼容、模型/维度/哈希失配 |
| 缓存预计算 | [tests/test_precompute_seeds.py](tests/test_precompute_seeds.py) | 2 | 部分失败不覆盖旧缓存、成功生成版本化缓存 |
| 启动诊断 | [tests/test_diagnostics.py](tests/test_diagnostics.py) | 5 | 模型映射、种子缓存、embedding LRU 容量、URL 与密钥脱敏 |
| 导入副作用 | [tests/test_import_side_effects.py](tests/test_import_side_effects.py) | 1 | 导入 config/main 不初始化运行时或创建目录 |
| 工具安全边界 | [tests/test_tool_safety.py](tests/test_tool_safety.py) | 20 | 结构化结果、一次性审批、无隐式 shell、环境隔离、保护路径与原子写入 |
| Embedding 服务 | [tests/test_embedding_service.py](tests/test_embedding_service.py) | 6 | 文本规范化、有界 LRU、淘汰、失败重试、并发请求合并与 Router 复用 |
| 压缩一致性 | [tests/test_router_compression.py](tests/test_router_compression.py) | 12 | 响应 schema、数量/去重/维度校验、原子替换、并发到达与陈旧快照保护 |
| 路由评测 | [tests/test_router_evaluation.py](tests/test_router_evaluation.py) | 5 | 数据集 schema、代理指标、误降级/误升级与风险加权阈值比较 |

**运行环境要求：**
- 不需要启动任何外部服务（Ollama、LiteLLM 均不需要）
- 只需要 Python 环境已安装 `pytest` 和项目依赖

### 6.2 性能基准测试

单独的性能基准脚本 [benchmark.py](benchmark.py)，输出格式化表格并导出 JSON 文件。

#### 6.2.1 仅本地模式（推荐，无需任何外部服务）

```bash
python benchmark.py
```

**测试内容：** 路由初始化耗时、128d/256d/512d/768d 余弦相似度计算速度、批量匹配、路由决策延迟（mock 嵌入）、错题本规模影响。

**输出位置：** 项目根目录下生成 `benchmark_results.json`，包含所有测试项的精确数值和单位。

> 本地模式下 embedding 调用被 mock，只测量路由逻辑的纯计算开销，排除 Ollama 网络波动干扰。初始化耗时约 700 µs，路由决策约 400 µs。

#### 6.2.2 含 API 模式（需 LiteLLM 代理在线）

```bash
python benchmark.py --api
```

**前置条件：** 必须先启动 LiteLLM 代理（参见第 3 节）。

**额外测试内容：** 在本地测试的基础上，新增：
- small 模型（deepseek-chat）真实 API 响应时间
- large 模型（deepseek-v4-pro）真实 API 响应时间
- 路由决策真实开销（含 Ollama 嵌入调用）

**输出位置：** 同样写入 `benchmark_results.json`，包含 API 延迟数据。

> 冷查询的路由耗时会被 Ollama `/api/embeddings` 调用主导。相同的规范化 query 会命中有界 LRU，不再重复请求 Ollama；容量由 `EMBEDDING_CACHE_SIZE` 控制，设为 `0` 可关闭缓存。并发的相同 query 也会合并为一次实际请求。

#### 6.2.3 输出文件说明

| 文件 | 由哪个命令生成 | 内容 |
|------|--------------|------|
| `benchmark_results.json` | `python benchmark.py [--api]` | 所有测试项的 name/value/unit/detail，JSON 数组格式 |
| `mistakes.json` | 路由测试写入（测试用后自动清理） | 错题本临时文件 |

`benchmark_results.json` 示例字段：

```json
[
  { "name": "路由初始化总耗时", "value": 0.000742, "unit": "秒", "detail": "81 条种子向量" },
  { "name": "small (deepseek-chat)", "value": 0.736, "unit": "秒", "detail": "3 次中位数: 789ms, 720ms, 736ms" },
  ...
]
```

### 6.3 离线路由评测

[routing_eval_cases.json](routing_eval_cases.json) 是版本化的人工标注自然语言样本，当前包含 24 条、small/large 各 12 条。[router_eval.py](router_eval.py) 可复现地统计准确率、误降级、误升级、估算成本与纯策略延迟，并比较一组候选阈值：

```bash
python router_eval.py
python router_eval.py --threshold 0.45 --json
```

默认离线词法代理在阈值 `0.45` 下得到 83.3% 准确率、0 次误降级、4 次误升级；风险加权比较建议 `0.40`。这里的 hashing embedding 仅用于 CI 回归趋势，不等同于生产环境语义 embedding，也不足以单独支持修改线上阈值。因此生产默认阈值仍保持 `0.45`；调整前应使用真实 Ollama embedding 和更大、来源明确的标注集复测。
