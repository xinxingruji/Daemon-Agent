"""预处理脚本：将 utterances.py 的种子文本调用 Ollama 转为向量并持久化到 seed_vectors.json。

运行一次即可：
    python precompute_seeds.py
"""

import json
import math
import os
import urllib.request
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from utterances import SMALL, LARGE

MODEL_NAME = "nomic-embed-text-v2-moe"
API_URL = "http://localhost:11434/api/embeddings"
OUTPUT = "seed_vectors.json"
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


def main():
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
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            status = "✓" if vec else "✗"
            print(f"\r  [{status}] 嵌入: |{bar}| {pct}% ({done}/{total})", end="", flush=True)
            if vec:
                result[route].append({"text": text, "vector": vec})

    print()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    small_count = len(result["small"])
    large_count = len(result["large"])
    print(f"[预处理] 完成！small: {small_count} 条, large: {large_count} 条 → {OUTPUT}")


if __name__ == "__main__":
    main()
