# SmallModel 改进方案：从 Mock 到真实模型 + 自我校验

## 问题诊断

**之前的 SmallModel**：
- 硬编码的规则匹配（keyword-based）
- 无法真实推理
- 只能通过固定规则触发低置信度
- 无法利用模型自身的"自我反思"能力

**现在的 SmallModel**：
- ✅ 支持真实的本地模型（Ollama）
- ✅ 自动降级到 Mock（Ollama 不可用）
- ✅ 检测模型自己表达的不自信
- ✅ 完整的向后兼容性

## 改进架构

```
┌─────────────────────────────────────────────────────────────┐
│                  SmallModel (Improved)                       │
├─────────────────────────────────────────────────────────────┤
│
│  优先级 1: 尝试真实模型 (Ollama)
│  ├─ OpenAI 兼容 API
│  ├─ 温度 0.3（更确切的回答）
│  └─ 最大 512 tokens
│
│  优先级 2: 自信度检测
│  ├─ 模型是否表达了不自信？
│  │  └─ "I am not sure", "I cannot fulfill", etc.
│  ├─ 答案是否太短？
│  │  └─ < 20 字符 + 长查询
│  └─ 返回标记 [UNCERTAIN_MODEL_OUTPUT]
│
│  优先级 3: 降级到 Mock
│  ├─ Ollama 不可用？
│  ├─ Ollama 调用失败？
│  └─ 使用规则匹配 + 启发式
│
│  输出：字符串（API 兼容）
│  └─ 可能包含 [UNCERTAIN_MODEL_OUTPUT] 标记
│
└─────────────────────────────────────────────────────────────┘

改进的 Evaluator
├─ 1. 检测模型标记的不自信
│  └─ [UNCERTAIN_MODEL_OUTPUT], [SHORT_RESPONSE], etc.
├─ 2. 传统长度/格式检查
└─ 3. LLM-as-Judge 最终判断
```

## 核心特性

### 1. 智能的 Ollama 支持 + Fallback

```python
small = SmallModel(
    use_ollama=True,              # 优先使用 Ollama
    model_name="qwen2.5:0.5b",    # 轻量级模型
    base_url="http://localhost:11434/v1",
    fallback_to_mock=True         # Ollama 失败时降级
)
```

**行为**：
- ✅ Ollama 可用 → 使用真实推理
- ✅ Ollama 不可用 → 自动降级到 Mock
- ✅ API 兼容性 100%（始终返回字符串）

### 2. 自我校验机制

模型生成答案后进行多维度检查：

```python
# 检查 1: 模型是否直接表达了不自信
if "I'm not sure" in response:
    return "[UNCERTAIN_MODEL_OUTPUT] " + response

# 检查 2: 答案长度异常
if len(response) < 20 and len(query) > 50:
    return "[SHORT_RESPONSE] " + response

# 检查 3: 通过标记信号给 Evaluator
```

### 3. 改进的 Evaluator

增强了三层检测：

```python
def evaluate(query, response):
    # 第1层：检测模型自己标记的不自信
    if "[UNCERTAIN_MODEL_OUTPUT]" in response:
        return "BAD"  # 立即升级
    
    # 第2层：传统检查（太短/为空）
    if len(response) < min_words:
        return "BAD"
    
    # 第3层：LLM 判官
    judge_result = llm_judge(query, response)
    return judge_result
```

## 使用场景对比

| 场景 | 旧方案 | 新方案 |
|------|--------|--------|
| **Ollama 可用** | 使用 Mock | ✅ 使用真实推理 |
| **模型表达不自信** | 等 Evaluator 检测 | ✅ 立即标记升级 |
| **Ollama 不可用** | N/A | ✅ 自动降级到 Mock |
| **API 兼容性** | ✅ Mock | ✅ 双模式兼容 |
| **可观测性** | 低 | ✅ 高（标记清晰） |

## 部署说明

### 方案 A：使用 Mock（开箱即用）

```python
small = SmallModel()  # 默认 use_ollama=True, fallback_to_mock=True
# 如果 Ollama 不可用，自动使用 Mock
```

### 方案 B：与 Ollama 集成（可选）

1. **安装 Ollama**：
   ```bash
   # macOS/Linux/Windows
   # 从 https://ollama.ai 下载安装
   ```

2. **下载轻量级模型**：
   ```bash
   ollama pull qwen2.5:0.5b      # ~350MB
   # 或
   ollama pull phi:2.7b           # ~1.5GB
   ```

3. **启动 Ollama 服务**：
   ```bash
   ollama serve  # 默认监听 http://localhost:11434
   ```

4. **Python 中使用**：
   ```python
   small = SmallModel(
       use_ollama=True,
       model_name="qwen2.5:0.5b"
   )
   ```

### 方案 C：禁用 Mock（只使用 Ollama）

```python
small = SmallModel(
    use_ollama=True,
    fallback_to_mock=False  # 如果 Ollama 不可用则抛出异常
)
```

## 执行流程示例

### 示例 1：简单查询（Ollama）

```
Query: "What is Python?"
  ↓
SmallModel (Ollama enabled)
  ├─ Call Ollama: "qwen2.5:0.5b"
  ├─ Response: "Python is a programming language..."
  ├─ Uncertainty check: PASS
  └─ Return: "Python is a programming language..."
  ↓
Evaluator
  ├─ Check uncertainty marker: NONE
  ├─ Check length: OK
  └─ Return: "GOOD"
```

### 示例 2：复杂查询（自信度降级）

```
Query: "Debug this distributed system"
  ↓
SmallModel (Ollama enabled)
  ├─ Call Ollama
  ├─ Response: "I'm not entirely confident..."
  ├─ Uncertainty check: FAIL (模型表达了不自信)
  └─ Return: "[UNCERTAIN_MODEL_OUTPUT] I'm not entirely..."
  ↓
Evaluator
  ├─ Check uncertainty marker: FOUND!
  └─ Return: "BAD" (立即升级)
```

### 示例 3：Ollama 不可用（降级到 Mock）

```
Query: "Debug this distributed system"
  ↓
SmallModel (Ollama enabled but unavailable)
  ├─ Try Ollama: FAIL (connection refused)
  ├─ Fallback to Mock
  ├─ Match keyword "debug": MATCH
  └─ Return: "I'm not entirely sure..."
  ↓
Evaluator
  ├─ Check uncertainty marker: NONE (未标记)
  ├─ Check length: OK
  └─ Return: "GOOD"
```

## 关键改进点

| 改进 | 实现 | 收益 |
|------|------|------|
| **真实推理** | Ollama 集成 | 更准确的回答 |
| **自我校验** | 模型自信度检测 | 更早的升级信号 |
| **韧性** | 自动 fallback | 生产可靠性 |
| **可观测性** | 清晰的标记 | 易于调试 |
| **兼容性** | 字符串返回 | 零 breaking changes |

## 配置建议

### 开发环境
```python
SmallModel(use_ollama=False)  # 使用 Mock，避免依赖
```

### 演示环境
```python
SmallModel(use_ollama=True, fallback_to_mock=True)
# 如果有 Ollama 就用，没有就用 Mock
```

### 生产环境
```python
SmallModel(
    use_ollama=True,
    fallback_to_mock=True,  # 保证可靠性
    model_name="qwen2.5:0.5b"  # 轻量高效
)
```

## 依赖说明

### 必需（已有）
- Python 3.8+
- 系统已有的所有包

### 可选（用于 Ollama）
```bash
pip install openai  # 仅用于 Ollama API 调用
```

### 如何检查 OpenAI 包状态
```python
try:
    from openai import OpenAI
    print("OpenAI 包可用")
except ImportError:
    print("OpenAI 包不可用，Ollama 集成将被禁用")
```

## 迁移路径

### 不需要改动的部分
- ✅ RuntimeV
- ✅ Evaluator（已更新以支持新标记）
- ✅ ModelManager
- ✅ 所有其他模块

### 需要更新的部分
- 🔄 SmallModel（已完成）
- 🔄 Evaluator（已完成）

### 验证方法
```bash
# 运行现有的 main.py，应该正常工作
python main.py --query "How to build adaptive inference runtime?"

# 如果有 Ollama，会使用真实模型；否则使用 Mock
```

## 总结

**这个改进方案的优势**：

1. ✅ **渐进式升级**：不需要一次性迁移到 Ollama
2. ✅ **自动降级**：失败时自动回到 Mock
3. ✅ **智能校验**：利用模型自己的不自信信号
4. ✅ **完全兼容**：现有代码无需修改
5. ✅ **生产就绪**：支持开发→演示→生产的全流程

**建议**：
- 开发阶段：继续使用 Mock（无 Ollama 依赖）
- 演示阶段：安装 Ollama，获得更好的演示效果
- 生产部署：使用轻量级模型（qwen2.5:0.5b）以平衡性能和质量
