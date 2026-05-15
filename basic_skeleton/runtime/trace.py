import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class Trace:
    COLORS = {
        "ROUTER": "\033[36m",        # Cyan
        "EXECUTION": "\033[34m",     # Blue
        "EVALUATOR": "\033[33m",     # Yellow
        "ESCALATION": "\033[35m",    # Magenta
        "MEMORY": "\033[32m",        # Green
        "RUNTIME": "\033[37m",       # White
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
    }

    def __init__(self, enabled: bool = True, use_colors: bool = True) -> None:
        self.enabled = enabled
        self.use_colors = use_colors
        self.history: List[Dict[str, Any]] = []

    def log(self, stage: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return

        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
        
        color = self.COLORS.get(stage, "")
        reset = self.COLORS["RESET"] if self.use_colors else ""
        
        line = f"[{color}{stage}{reset}] {timestamp} | {message}"
        if meta:
            line += " | " + json.dumps(meta, ensure_ascii=True)
        
        print(line)
        
        self.history.append({
            "timestamp": timestamp,
            "stage": stage,
            "message": message,
            "meta": meta or {},
        })

    def print_runtime_summary(self) -> None:
        if not self.history:
            print("\n[SUMMARY] No trace history recorded.")
            return

        print("\n" + "=" * 70)
        print(f"{self.COLORS['BOLD']}RUNTIME SUMMARY{self.COLORS['RESET']}")
        print("=" * 70)

        stage_counts = {}
        for entry in self.history:
            stage = entry["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        print(f"\n{self.COLORS['BOLD']}Stage Breakdown:{self.COLORS['RESET']}")
        for stage in ["ROUTER", "EXECUTION", "EVALUATOR", "ESCALATION", "MEMORY", "RUNTIME"]:
            if stage in stage_counts:
                count = stage_counts[stage]
                color = self.COLORS.get(stage, "")
                print(f"  {color}[{stage}]{self.COLORS['RESET']} {count} event(s)")

        print(f"\n{self.COLORS['BOLD']}Event Log:{self.COLORS['RESET']}")
        for i, entry in enumerate(self.history, 1):
            stage = entry["stage"]
            timestamp = entry["timestamp"]
            message = entry["message"]
            color = self.COLORS.get(stage, "")
            print(f"  {i}. {color}[{stage}]{self.COLORS['RESET']} {timestamp} | {message}")
            if entry["meta"]:
                meta_str = json.dumps(entry["meta"], ensure_ascii=True)
                print(f"     └─ {meta_str}")

        print("=" * 70 + "\n")

    def clear_history(self) -> None:
        self.history = []

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()
