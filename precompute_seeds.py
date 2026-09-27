"""预处理脚本：将 utterances.py 的种子文本调用 Ollama 转为向量并持久化到 seed_vectors.json。

运行一次即可：
    python precompute_seeds.py
"""

import json
import os
import urllib.request
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from utterances import SMALL, LARGE
from seed_cache import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    build_seed_cache_document,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

MODEL_NAME = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
API_URL = os.getenv("OLLAMA_EMBEDDING_URL", DEFAULT_EMBEDDING_URL)
OUTPUT = Path("seed_vectors.json")
MAX_WORKERS = 8


def _get_embedding(text: str) -> List[float]:
    payload = {"model": MODEL_NAME, "prompt": text}
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))["embedding"]
    except Exception as e:
        print(f"  [错误] 嵌入失败: '{text}' -> {e}", file=sys.stderr)
        return []


def main() -> int:
    global MODEL_NAME, API_URL
    if load_dotenv is not None:
        load_dotenv(override=True)
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    API_URL = os.getenv("OLLAMA_EMBEDDING_URL", DEFAULT_EMBEDDING_URL)

    print(f"[预处理] 开始将种子文本转为向量（{MAX_WORKERS} 线程并发）...")
    print(f"[预处理] 嵌入模型: {MODEL_NAME}")

    result: Dict[str, list] = {"small": [], "large": []}
    tasks = []

    for route_name, utterances in [("small", SMALL), ("large", LARGE)]:
        for text in utterances:
            tasks.append((route_name, text))

    total = len(tasks)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_get_embedding, text): (route, text)
            for route, text in tasks
        }
        for future in as_completed(future_map):
            route, text = future_map[future]
            vec = future.result()
            done += 1
            pct = done * 100 // total
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            status = "OK" if vec else "FAIL"
            print(f"\r  [{status}] 嵌入: |{bar}| {pct}% ({done}/{total})", end="", flush=True)
            if vec:
                result[route].append({"text": text, "vector": vec})

    print()

    small_count = len(result["small"])
    large_count = len(result["large"])
    if small_count != len(SMALL) or large_count != len(LARGE):
        print(
            "[预处理] 失败：部分种子未生成向量，保留现有缓存不变。",
            file=sys.stderr,
        )
        return 1

    document = build_seed_cache_document(
        result,
        embedding_model=MODEL_NAME,
        source_routes={"small": SMALL, "large": LARGE},
    )
    temp_output = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with open(temp_output, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    temp_output.replace(OUTPUT)

    print(f"[预处理] 完成！small: {small_count} 条, large: {large_count} 条 → {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
