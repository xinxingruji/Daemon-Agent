# 种子库动态进化系统 — 修改记录

## 改动文件一览

| 文件 | 操作 | 说明 |
|------|------|------|
| `precompute_seeds.py` | 新建 | 预处理脚本，8 线程并发调 Ollama 生成 seed_vectors.json |
| `router.py` | 修改 | 启动读缓存、新增种子管理方法、route() 记录反馈信息 |
| `main.py` | 修改 | 工具执行块加反馈逻辑、REPL 加 /reload 命令 |
| `seed_vectors.json` | 生成 | 82 条种子向量的持久化缓存文件（1.8 MB） |
| `utterances.py` | 修改 | 删掉 SMALL 中重复的 "查看" |
| `tests/conftest.py` | 修改 | mock 向量 3→768 维、防磁盘写入、注入测试种子、移入 --run-api |
| `tests/test_router.py` | 修改 | 新功能测试 6 个、清理无用临时文件 |
| `tests/test_benchmarks.py` | 修改 | 维度对齐、准确率修复、API 多次测量、客户端初始化修复 |
| `benchmark.py` | 修改 | 维度 128→768、API 多次测量取中位数、client 清理 AUTH_TOKEN、large 标签更新 |
| `litellm_config.yaml` | 修改 | large 改用 deepseek-v4-pro |
| `litellm_config.example.yaml` | 修改 | large 改用 deepseek-v4-pro |

---

## 1. precompute_seeds.py（新建）

**作用：** 一次性预处理，把 utterances.py 的种子文本转成向量存入 seed_vectors.json。
**运行方式：** `python precompute_seeds.py`
**关键设计：**
- 8 线程 ThreadPoolExecutor 并发调 Ollama
- 82 条嵌入 ~10 秒完成（原来串行 ~80 秒）
- 输出格式：`{"small": [{"text":"...","vector":[...]},...], "large": [...]}`
- 输入来源：`utterances.py` 的 SMALL 和 LARGE 列表

---

## 2. router.py 修改

### 2.1 __init__ 改动

**原来：** 调用 SMALL/LARGE 列表，逐条串行调 Ollama 嵌入，82 条 ~80 秒。
**现在：** 优先读 seed_vectors.json（毫秒级），不存在时回退到原来的 Ollama 方式并自动写出缓存。

新增参数：`seed_file: str = "seed_vectors.json"`
新增属性：`route_embeddings_text`（存储种子文本，与向量一一对应）
新增属性：`_last_query_vector`、`_last_route_scores`、`_last_best_route`（供反馈用）

### 2.2 新增方法

```python
_load_seed_vectors()      # 从 JSON 加载预计算向量；不存在则回退
_save_seed_vectors()      # 将当前种子库写入 JSON
add_seed(text, route)     # 添加新种子（调 Ollama 嵌入 + 写文件）
remove_most_similar_seed(vec, route)  # 删除与查询向量最相似的种子
reload_seeds()            # 热重载 seed_vectors.json
```

### 2.3 route() 改动

新增 3 行记录：
- `self._last_query_vector = query_vector`
- `self._last_route_scores[route_name] = 每路最高分`
- `self._last_best_route = best_route`

路由决策逻辑本身未变。

### 2.4 修复

- 惩罚打印条件：`if penalty > 0` 才打印上调信息（之前 penalty=0 也会打印 "已从 0.45 上调至 0.450"）

---

## 3. main.py 修改

### 3.1 种子库动态反馈（agent_loop 工具执行块）

```
小模型工具调用失败 → record_mistake（原有）+ remove_most_similar_seed（新增）
小模型工具调用成功 + 匹配分 < threshold + 0.15 → add_seed（新增）
```

### 3.2 /reload 命令（REPL 循环）

```
/reload → ROUTER.reload_seeds() → 热重载 seed_vectors.json
```

### 3.3 强制模型前缀（REPL 循环）

```
!large 查询内容  → 跳过路由，强制使用大模型
!small 查询内容  → 跳过路由，强制使用小模型
查询内容         → 正常走三阶段路由
```

实现：REPL 检测 `!large ` / `!small ` 前缀，剥离前缀后传入 `agent_loop(force_mode=...)`。

### 3.4 agent_loop 签名改动

```python
# 原来
def agent_loop(messages: list, query: str):
# 现在
def agent_loop(messages: list, query: str, force_mode: str = ""):
# force_mode: "" = 跟随系统, "large" = 强制大模型, "small" = 强制小模型
```

---

## 4. router.py route() 签名改动

```python
# 原来
def route(self, query, total_tokens=0, force_large=False) -> str:
# 现在
def route(self, query, total_tokens=0, force_large=False, force_small=False) -> str:
```

新增 `force_small` 参数，为 True 时直接返回 "small"，跳过所有路由判断。

---

## 5. 测试文件修改

### 5.1 tests/conftest.py

**原来：** mock 向量 3 维 `[0.5, 0.5, 0.5]`，测试中 `_save_seed_vectors` 会在项目目录写入真实文件。

**现在：**
- mock 向量改为 768 维 `[0.5] * 768`，匹配真实 Ollama 输出维度
- 新增 `_save_seed_vectors` mock（空操作），防止测试写磁盘
- 新增 `_load_seed_vectors` mock（`_fake_load_seeds`），注入 small=5、large=3 的测试种子，不读磁盘
- 移入 `pytest_addoption("--run-api")` 选项（原来写在 test_benchmarks.py 里不生效）

### 5.2 tests/test_router.py — 新增 6 个测试

| 测试 | 验证点 |
|------|--------|
| `test_force_small` | `force_small=True` 直接返回 "small" |
| `test_add_seed` | 添加种子后文本和向量同步追加 |
| `test_add_seed_no_duplicate` | 重复文本不被重复添加 |
| `test_add_seed_empty_skip` | 空文本跳过不添加 |
| `test_remove_most_similar_seed` | 注入已知向量，删掉余弦距离最近的那条 |
| `test_reload_seeds` | 临时文件写入 → 修改内存 → reload → 验证从文件恢复 |

**清理：** `test_mistake_intercepted` 删掉无用的 `tempfile.NamedTemporaryFile`，改为手动设置 `mistake_book`。

测试总数从 11 → 17。

### 5.3 tests/test_benchmarks.py

**维度对齐：** 所有 mock `[0.5] * 128` → `[0.5] * 768`。

**准确率修复：** 原来所有查询返回相同 mock 向量，准确率测试无意义。现在：
- small 种子 = `[1.0, 0.0, 0.0, ...]`，large 种子 = `[0.0, 1.0, 0.0, ...]`
- 期望 small 的查询 mock 为 `[0.9, 0.1, 0.0, ...]`（接近 small 种子）
- 期望 large 的查询 mock 为 `[0.1, 0.9, 0.0, ...]`（接近 large 种子）
- 两方向余弦距离完全分离，准确率可达到 100%

**API 延迟多次测量：** `test_small_model_latency` / `test_large_model_latency` 从单次 → 3 次取中位数，输出每次耗时 + 中位数。

**客户端初始化修复：** `real_client` fixture 对齐 `config.py`：
- `load_dotenv(override=True)` 覆盖继承的环境变量
- `os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)` 清除父进程注入的 token
- 防止请求被路由到错误的 API 端点

### 5.4 utterances.py

删掉 SMALL 中重复的 "查看"（原在第 7 行和第 12 行各出现一次）。修复后 `test_small_no_duplicates` 通过。

---

## 使用流程

首次使用：
    python precompute_seeds.py          # 生成 seed_vectors.json（一次性）
    python main.py                      # 秒启，直接读缓存

日常使用中编辑种子：
    手动编辑 seed_vectors.json
    Daemon >> /reload                  # 立即生效，不重启

运行时自动进化：
    小模型失败 → 自动删除误导的种子
    小模型成功但勉强 → 自动添加查询为新种子

---

## 未改动的部分

- `_load_mistakes` / `record_mistake` — 错题本逻辑未动
- `_get_embedding` / `_cosine_similarity` — 嵌入和计算未动
- `route()` 决策核心 — 三阶段（错题本 → 语义匹配 → token 惩罚）未动
- `agent_loop` 循环结构 — 未动
- `core_tools.py` — 未动
- `managers.py` / `team.py` / `config.py` — 未动
