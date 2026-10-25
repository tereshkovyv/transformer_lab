"""Run every tests/check_*.py in its own process and print a PASS/FAIL summary."""
import os
import subprocess
import sys
from pathlib import Path

scripts = sorted(Path(__file__).parent.glob("check_*.py"))
failed = []
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # generated text may contain non-cp1252 characters on Windows
for script in scripts:
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8", env=env)
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'}  {script.name}")
    if result.returncode != 0:
        failed.append(script.name)
        print(result.stdout[-1500:] + result.stderr[-1500:])

print(f"\n{len(scripts) - len(failed)}/{len(scripts)} passed")
sys.exit(1 if failed else 0)
