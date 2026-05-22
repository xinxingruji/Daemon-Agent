# 🚀 Auto-Adaptive Agentic Framework (自适应多智能体系统)

本项目基于极简的 `learn_claude_code` 架构，进行了一次深度的工业级重构。我们为其注入了“成本感知”、“风险感知”**与**“群体记忆”能力，打造了一个完全摆脱对单一昂贵大模型依赖的智能调度中枢（Router）。

## 🧬 基座框架简介: `learn_claude_code`

本系统的底层运行逻辑脱胎于经典的 `s_full.py` 架构，这是一个纯手写的、无外部沉重依赖（无 LangChain / AutoGen 等沉重包袱）的轻量级多智能体基座：

* **ReAct 循环**：以 `Thought -> Action -> Observation` 为核心的执行引擎。
* **Plan-and-Solve (Todo 机制)**：主脑（Lead）通过严格的 Todo 列表管理多步复杂任务。
* **多智能体拓扑**：支持主脑（Lead）、持续工作的队友（Teammate）和短生命周期的探索小组（Subagent）之间的异步通信（Message Bus）与任务抢占。

---

## 🌟 核心革新：自适应微观路由 (Auto-Adaptive Micro-Routing)

原版框架在运行时会全局锁死单一模型（通常是极其昂贵的如 Claude 3.5 Sonnet 等）。为了实现极致的降本增效与防崩溃，我们为其加装了名为 **`Claude_Router`** 的独立“小脑”。

### 1. 🔀 宏观与微观双轨调度 (Dual-Track Routing)

传统的 Agent 往往“高射炮打蚊子”，我们彻底改变了这一现状：

* **大门宏观拦截 (REPL 级)**：当用户下达指令时，路由器首先对 Query 进行语义测算。如果是重构架构级别的超级难题，全局直接锁定大模型 (`force_large=True`)。
* **细胞级微观嗅探 (Loop 级)**：如果判定为常规任务，系统将默认放行给小模型（如 Qwen-7B 等高性价比模型）。在随后的成百上千次 ReAct 循环中，系统通过 **“意图嗅探器 (Intent Sniffer)”**，实时判断 Agent 当下到底在干什么：
* **主脑**：嗅探当前的 `Todo.items`。
* **队友**：逆向解析最近的 `<auto-claimed>` (任务板) 或 `<inbox>` (主帅私信) 标签。
* **效果**：小模型如果当前只是在执行 `ls` 或读取日志，系统自动分配极低成本模型；如果需要重写核心算法，系统瞬间请求高算力大模型接管。



### 2. 🛡️ 智能错题本与群体免疫 (Self-Healing Mistake Book)

小模型的执行极其容易翻车，我们设计了一套**滴水不漏的监控网**：

* **智能错误边界**：重构了底层的工具契约（Tool Contract）。对于 `grep`, `diff` 等探测型 Bash 命令的非 0 退出码予以豁免；对于真正的崩溃和 JSON 格式错误，打上 `Error:` 标签。
* **动态刻录**：一旦小模型翻车，系统不仅会拦截，还会把“导致翻车的当前微观任务 + 崩溃的工具名”记录到持久化的本地向量错题本中。
* **容量控制与去重**：采用 FIFO (先进先出) 机制淘汰旧数据，并进行 `> 0.98` 的极高相似度去重，确保本地错题本文件极度精简，避免检索时的线性性能衰减。
* **群体共享**：主脑踩过的坑，子智能体绝不会再踩。下一次遇到相似度 `>= 0.85` 的任务，路由器会直接拉响警报，在任务执行前强制升维大模型。

### 3. 📉 长上下文衰减防御 (Context Degradation Penalty)

长文本会导致小模型出现严重的指令遗忘（Lost in the middle）。我们引入了极其极客的**阶梯式 Token 惩罚算法**：

* 系统实时监控 `messages` 的总 Token 数量。
* 设定安全区（如 3000 Tokens）和步长（如 4000 Tokens）。
* 每当上下文溢出一个步长，小模型继续执行任务的“及格线 (Threshold)”就会被硬性上调（例如从 0.75 阶梯上涨至 0.80）。当报错或上下文堆积如山时，系统不再信任小模型，自动移交大模型清场。

---

## ⚙️ 代码实现亮点 (Implementation Details)

1. **零依赖向量引擎**：路由器不依赖任何庞大的第三方库，通过简单的 JSON HTTP 请求调用本地 Ollama（`nomic-embed-text`）完成极速的向量余弦相似度计算，耗时在毫秒级，完全不拖慢主循环。
2. **纯净的降噪解析**：在处理 Teammate 的 Inbox 私信时，巧妙结合了 XML 标签（`<inbox>`）与 JSON 反序列化，剔除了底层控制指令的语义噪音，将最纯粹的用户意图剥离出来喂给路由器，确保了余弦计算的 100% 精准。
3. **无缝热插拔**：路由机制对原版 `s_full.py` 的破坏性极小，仅在模型 API 调用的前置切面进行了拦截替换，完美遵守了开闭原则（Open-Closed Principle）。

---

## 🚀 如何运行 (Usage)

本系统依赖 **Ollama**（提供本地向量嵌入）和 **LiteLLM Proxy**（统一大小模型 API 调用网关）来运行。

### 1. 环境准备与依赖安装

克隆项目后，首先安装所需的 Python 依赖包：

```bash
pip install -r requirements.txt

```

### 2. 启动本地向量大脑 (Ollama)

我们需要启动 Ollama 服务，并拉取用于计算语义相似度的轻量级嵌入模型：

```bash
# 启动 Ollama (如果在后台运行则跳过)
ollama serve

# 拉取 nomic-embed-text 模型
ollama run nomic-embed-text-v2-moe

```

### 3. 配置并启动 LiteLLM 模型网关

我们的代码中动态路由了 `small` 和 `large` 两个模型名称。需要通过 LiteLLM 将它们映射到真实的模型（比如本地的 Qwen 和线上的 Claude）。

在项目根目录创建一个 `litellm_config.yaml` 文件，示例如下：

```yaml
model_list:
  - model_name: small
    litellm_params:
      # 替换为你实际部署的小模型 (如 ollama/qwen2.5:7b)
      model: ollama/qwen2.5:7b 
  - model_name: large
    litellm_params:
      # 替换为你实际使用的大模型 (如 claude-3-5-sonnet-20241022 或 deepseek-chat)
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

```

配置好 `.env` 文件中的 API Key 后，**在一个新终端窗口启动 LiteLLM 代理**：

```bash
litellm --config litellm_config.yaml --port 4000

```

### 4. 配置环境变量并启动主程序

先复制一份环境配置模板：

```bash
cp .env.example .env

```

然后打开 `.env` 文件，进行如下配置：

1. 配置 Anthropic SDK 指向刚才启动的 LiteLLM 代理：
```env
ANTHROPIC_BASE_URL="http://0.0.0.0:4000"

```


2. 填入你实际使用的云端大模型接口的 API Key：
```env
ANTHROPIC_API_KEY="your-api-key-here"

```


3. *(可选)* 填入云端大模型的具体名称：
```env
MODEL_ID="your-cloud-model-name" 

```



**一切就绪！现在，运行主程序：**

```bash
python main.py

```

*(注：如果你的主入口文件仍然叫 `s_full.py`，请替换为 `python s_full.py`)*

进入 `s_full >>` 终端后，输入你的任务，尽情感受智能体在小模型与大模型之间丝滑切换的技术美学吧！

---

## 开源与贡献

本项目当前采用 [MIT License](LICENSE)。如果你准备发起改动，建议先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 里的贡献流程、代码规范和社区维护约定。

后续如果要进一步完善社区规范，建议继续补充：

- `CODE_OF_CONDUCT.md`：明确社区行为准则。
- `CHANGELOG.md`：记录版本演进和兼容性变化。
- `ISSUE_TEMPLATE` / `PULL_REQUEST_TEMPLATE`：统一问题反馈和 PR 描述格式。
