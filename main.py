# 这里实例化所有的模块、定义全局 Tools Schema、注入回调，并运行控制台的主循环。

import json
import sys
import uuid
from config import client, WORKDIR, SKILLS_DIR, TOKEN_THRESHOLD, VALID_MSG_TYPES, ROUTER
from core_tools import run_bash, run_read, run_write, run_edit, estimate_tokens, microcompact, is_tool_error
from managers import TodoManager, SkillLoader, TaskManager, BackgroundManager, MessageBus
from team import TeammateManager, run_subagent, auto_compact

# 重配 stdout 编码，防止 UTF-8 内容打印到 GBK 终端时 UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# === SECTION: global_instances ===
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)

# === SECTION: system_prompt ===
SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
Prefer task_create/task_update/task_list for multi-step work. Use TodoWrite for short checklists.
Use task for subagent delegation. Use load_skill for specialized knowledge.
Skills: {SKILLS.descriptions()}"""

# === SECTION: shutdown + plan tracking (s10) ===
shutdown_requests = {}
plan_requests = {}

# === SECTION: shutdown_protocol (s10) ===
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"

# === SECTION: plan_approval (s10) ===
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = plan_requests.get(request_id)
    if not req: return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"

# === SECTION: tool_dispatch (s02) ===
TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "TodoWrite":        lambda **kw: TODO.update(kw["items"]),
    "task":             lambda **kw: run_subagent(kw["prompt"], kw.get("agent_type", "Explore")),
    "load_skill":       lambda **kw: SKILLS.load(kw["name"]),
    "compress":         lambda **kw: "Compressing...",
    "background_run":   lambda **kw: BG.run(kw["command"], kw.get("timeout", 120)),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "task_create":      lambda **kw: TASK_MGR.create(kw["subject"], kw.get("description", "")),
    "task_get":         lambda **kw: TASK_MGR.get(kw["task_id"]),
    "task_update":      lambda **kw: TASK_MGR.update(kw["task_id"], kw.get("status"), kw.get("add_blocked_by"), kw.get("remove_blocked_by")),
    "task_list":        lambda **kw: TASK_MGR.list_all(),
    "spawn_teammate":   lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":   lambda **kw: TEAM.list_all(),
    "send_message":     lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "plan_approval":    lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":             lambda **kw: "Lead does not idle.",
    "claim_task":       lambda **kw: TASK_MGR.claim(kw["task_id"], "lead"),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "TodoWrite", "description": "Update task tracking list.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["content", "status", "activeForm"]}}}, "required": ["items"]}},
    {"name": "task", "description": "Spawn a subagent for isolated exploration or work.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": ["Explore", "general-purpose"]}}, "required": ["prompt"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compress", "description": "Manually compress conversation context.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "background_run", "description": "Run command in background thread.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "task_create", "description": "Create a persistent file task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_get", "description": "Get task details by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "task_update", "description": "Update task status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}, "add_blocked_by": {"type": "array", "items": {"type": "integer"}}, "remove_blocked_by": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "spawn_teammate", "description": "Spawn a persistent autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]

def is_tool_error(output: str) -> bool:
    out_str = str(output).strip()
    # 这是针对run_bash的，如果返回码非0，那一定出错了
    # 绝对确定的系统级错误前缀 
    error_prefixes = (
        "Error:", 
        "error:", 
        "ERROR:",
        "Unknown",        # 捕获 "Unknown tool: xxx"
        "KeyError",       # 防御性捕获
        "Exception",      # 防御性捕获
        "Traceback",      # 捕获未格式化的 Python 崩溃堆栈
        "Fatal:",         # 捕获一些底层库的致命错误
        "Trace/BPT trap"  # 捕获 C 级别底层崩溃
    )
    if out_str.startswith(error_prefixes):
        return True
        
    # 特定工具的隐式失败标志
    if out_str in ("(subagent failed)", "Unknown tool", "(no summary)"):
        return True
        
    return False

# === SECTION: agent_loop ===
def agent_loop(messages: list, query: str, force_mode: str = ""):
    """force_mode: "" = 跟随系统, "large" = 强制大模型, "small" = 强制小模型"""
    rounds_without_todo = 0
    while True:
        # s06: compression pipeline
        microcompact(messages)
        current_tokens = estimate_tokens(messages)
        if current_tokens > TOKEN_THRESHOLD:
            print("[auto-compact triggered]")
            messages[:] = auto_compact(messages)
        # s08: drain background notifications
        notifs = BG.drain()
        if notifs:
            txt = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append({"role": "user", "content": f"<background-results>\n{txt}\n</background-results>"})
        # s10: check lead inbox
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})

# 路由
        current_subtask = ""
        if TODO.items:
            # 优先寻找正在进行中的任务
            for item in TODO.items:
                if item["status"] == "in_progress":
                    current_subtask = item["content"]
                    break
            # 如果没有 in_progress，就拿第一个排队的任务
            if not current_subtask:
                for item in TODO.items:
                    if item["status"] == "pending":
                        current_subtask = item["content"]
                        break
        
        routing_query = current_subtask if current_subtask else query
        current_model = ROUTER.route(
            query=routing_query, total_tokens=current_tokens,
            force_large=(force_mode == "large"),
            force_small=(force_mode == "small"),
        )
            
# 路由
        # LLM call
        response = client.messages.create(
            model=current_model, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        # Tool execution
        results = []
        used_todo = False
        manual_compress = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compress":
                    manual_compress = True
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                out_str = str(output)
                # 按工具类型定制显示，避免无差别 dump 内容
                if block.name == "bash":
                    print(f"\033[36m> bash:\033[0m")
                    print(out_str[:200])
                elif block.name == "read_file":
                    path = block.input.get("path", "").replace(str(WORKDIR), ".")
                    lines = out_str.count('\n') if not out_str.startswith("Error:") else 0
                    if out_str.startswith("Error:"):
                        print(f"\033[31m> read_file: {out_str[:120]}\033[0m")
                    else:
                        print(f"\033[34m> 📄 read_file: {path} ({lines} 行)\033[0m")
                elif block.name in ("write_file", "edit_file"):
                    path = block.input.get("path", "").replace(str(WORKDIR), ".")
                    print(f"\033[32m> ✏️ {block.name}: {path}\033[0m")
                elif block.name == "TodoWrite":
                    items = block.input.get("items", [])
                    summary = ", ".join(f"{i.get('status','?')}:{i.get('content','')[:30]}" for i in items)
                    print(f"\033[33m> 📋 TodoWrite: {summary[:150]}\033[0m")
                else:
                    print(f"> {block.name}:")
                    print(out_str[:200])

                # 记录错题本
                if is_tool_error(output) and current_model == "small":
                    ROUTER.record_mistake(routing_query)

                # 小模型成功但匹配分低 → 把查询加入种子库
                if not is_tool_error(output) and current_model == "small":
                    small_score = ROUTER._last_route_scores.get("small", 0.0)
                    if 0 < small_score < ROUTER.threshold + 0.15:
                        ROUTER.add_seed(routing_query, "small")

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "TodoWrite":
                    used_todo = True
        # s03: nag reminder (only when todo workflow is active)
        rounds_without_todo = 0 if used_todo else rounds_without_todo + 1
        if TODO.has_open_items() and rounds_without_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})
        # s06: manual compress
        if manual_compress:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


# === SECTION: repl ===
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36mDaemon >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/compact":
            if history:
                print("[manual compact via /compact]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/tasks":
            print(TASK_MGR.list_all())
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        if query.strip() == "/reload":
            ROUTER.reload_seeds()
            continue
        # ── 处理强制模型前缀 ──
        force_mode = ""
        if query.startswith("!large "):
            force_mode = "large"
            query = query[7:]
            print(f"\033[33m已强制使用: large\033[0m")
        elif query.startswith("!small "):
            force_mode = "small"
            query = query[7:]
            print(f"\033[33m已强制使用: small\033[0m")

        history.append({"role": "user", "content": query})
        agent_loop(history, query, force_mode=force_mode)

        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
