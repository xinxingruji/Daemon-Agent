import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_importing_main_does_not_initialize_runtime_or_create_directories(tmp_path):
    script = """
import json
from pathlib import Path
import config
import main
print(json.dumps(config.runtime_initialization_status(), sort_keys=True))
print(json.dumps(sorted(item.name for item in Path.cwd().iterdir())))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")],
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = completed.stdout.strip().splitlines()

    assert json.loads(lines[-2]) == {
        "client": False,
        "environment": False,
        "router": False,
    }
    assert json.loads(lines[-1]) == []
