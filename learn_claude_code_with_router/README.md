# 🚀 如何运行 (Usage)

本系统依赖 **Ollama**（提供本地向量嵌入）和 **LiteLLM Proxy**（统一大小模型 API 调用网关）来运行。

---

## 1. 环境准备与依赖安装

克隆项目后，首先安装所需的 Python 依赖包：

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
ollama run nomic-embed-text

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
> 这样可以关闭本地模型的思考模式，本地模型还是不要思考了，思考半天屁都放不出来一个。api_base: http://localhost:11434 貌似可以去掉

配置好后**在一个新终端窗口启动 LiteLLM 代理**：

```bash
litellm --config litellm_config.yaml --port 4000

```

---

## 4. 环境配置与启动主程序

### 环境变量配置

先复制一份环境配置模板：

```bash
cp .env.example .env

```

然后打开 `.env` 文件，进行如下配置。配置 Anthropic SDK 指向刚才启动的 LiteLLM 代理：

```env
ANTHROPIC_BASE_URL="http://localhost:4000"

```

### 运行主程序

**一切就绪！现在，运行主程序：**

```bash
python main.py

```

*(注：如果你的主入口文件仍然叫 `s_full.py`，请替换为 `python s_full.py`)*

进入 `s_full >>` 终端后，输入你的任务，尽情感受智能体在小模型与大模型之间丝滑切换的技术美学吧！

```

```
