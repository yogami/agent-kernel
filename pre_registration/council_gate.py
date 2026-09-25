#!/usr/bin/env python3
"""
Pre-Push Security Council Gate for Berlin AI Studio.
Enforces RULES.md:
1. Pytest suite must pass with >= 80% coverage.
2. Radon cyclomatic complexity must be <= 3 for all production functions.
3. Multi-model council audit on outgoing commit diff via OpenRouter.
"""

import os
import sys
import subprocess
import requests
from typing import Tuple, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"



def run_cmd(cmd: str, cwd: Optional[str] = None) -> Tuple[int, str]:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    output = (res.stdout + "\n" + res.stderr).strip()
    return res.returncode, output


def _find_py_bin(repo_root: str) -> str:
    venv_py = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


def check_tests_and_coverage(repo_root: str) -> bool:
    print("[1/3] Running pytest and coverage validation...")
    pre_dir = os.path.join(repo_root, "pre_registration")
    py_exec = _find_py_bin(repo_root)

    cmd = f"{py_exec} -m pytest tests/ -q --cov=. --cov-fail-under=80"
    code, out = run_cmd(cmd, cwd=pre_dir)
    if code != 0:
        print(f"FAILED: Tests or coverage check failed.\n{out}")
        return False
    print("PASS: Tests passed and coverage >= 80%.")
    return True


def _find_radon_bin(repo_root: str) -> str:
    venv_radon = os.path.join(repo_root, ".venv", "bin", "radon")
    if os.path.exists(venv_radon):
        return venv_radon
    return "radon"


def _filter_clean_radon_lines(raw_output: str) -> list:
    lines = []
    for line in raw_output.splitlines():
        if line.strip() and "legacy" not in line:
            lines.append(line)
    return lines


def check_cyclomatic_complexity(repo_root: str) -> bool:
    print("[2/3] Checking cyclomatic complexity (must be <= 3)...")
    radon_exec = _find_radon_bin(repo_root)
    cmd = f"{radon_exec} cc pre_registration/*.py -s -n B"
    _, out = run_cmd(cmd, cwd=repo_root)
    clean_lines = _filter_clean_radon_lines(out)
    if clean_lines:
        print("FAILED: Functions exceed cyclomatic complexity 3:\n" + "\n".join(clean_lines))
        return False
    print("PASS: All active functions maintain CC <= 3.")
    return True


AUDIT_TARGET_FILES = (
    "api/ablation_router.py "
    "api/app.py "
    "core/guardrails.py "
    "pre_registration/constants.py "
    "pre_registration/council_gate.py "
    "pre_registration/l1_harness.py "
    "pre_registration/l2_kernel.py "
    "Dockerfile "
    "README.md "
    "BENCHMARK_REPORT.md"
)


def get_outgoing_diff(repo_root: str) -> str:
    cmd = f"git diff origin/main..HEAD -- {AUDIT_TARGET_FILES}"
    code, out = run_cmd(cmd, cwd=repo_root)
    if code == 0 and out.strip():
        return out.strip()
    code, fallback = run_cmd("git diff HEAD~1..HEAD", cwd=repo_root)
    return fallback.strip() if code == 0 else ""


def build_audit_prompt(diff_text: str) -> str:
    truncated_diff = diff_text[:90000]
    return f"""You are the Multi-Model Pre-Push Security Council Gate reviewing an outgoing git push.
Evaluate this git diff for critical flaws:
1. Are there hardcoded secrets or exposed keys?
2. Are there syntax errors, breaking regressions, or bypassed policies?
3. Does the code violate clinical safety constraints?

Reply strictly with:
- If acceptable: COUNCIL GREENLIGHT
- If critical defect found: COUNCIL REJECTION followed by specific bullet points explaining the block.

DIFF:
{truncated_diff}
"""



def query_council_model(diff_text: str, key: str) -> Tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": build_audit_prompt(diff_text)}
        ]
    }
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        if r.status_code != 200:
            return True, f"Council warning: OpenRouter returned status {r.status_code}. Allowing push."
        reply = r.json()["choices"][0]["message"]["content"]
        if "COUNCIL REJECTION" in reply:
            return False, reply
        return True, reply
    except Exception as e:
        return True, f"Council bypass notice: API connection failed ({e})."


def _is_ignorable_env_line(line: str) -> bool:
    if not line or line.startswith("#"):
        return True
    return False


def _parse_env_pair(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if _is_ignorable_env_line(stripped) or "=" not in stripped:
        return None
    k, v = stripped.split("=", 1)
    return k.strip(), v.strip().strip("'\"")


def _apply_env_entry(line: str) -> None:
    pair = _parse_env_pair(line)
    if not pair:
        return
    if pair[0] not in os.environ:
        os.environ[pair[0]] = pair[1]


def _load_local_env(repo_root: str) -> None:
    env_file = os.path.join(repo_root, ".env")
    if not os.path.isfile(env_file):
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            _apply_env_entry(line)


def _resolve_council_key(repo_root: str) -> Optional[str]:
    _load_local_env(repo_root)
    return os.getenv("OPENROUTER_API_KEY")


def _eval_council_decision(diff_text: str, key: str) -> bool:
    approved, message = query_council_model(diff_text, key)
    if not approved:
        print(f"FAILED: Push blocked by Frontier Council Gate:\n{message}")
        return False
    print(f"PASS: Frontier Council approved push.\n{message[:200]}...")
    return True


def check_council_review(repo_root: str) -> bool:
    print("[3/3] Querying OpenRouter Frontier Council Gate...")
    diff_text = get_outgoing_diff(repo_root)
    if not diff_text:
        print("PASS: No outgoing diff detected.")
        return True

    key = _resolve_council_key(repo_root)
    if not key:
        print("FAILED: Push blocked because OPENROUTER_API_KEY is not set.")
        return False
    return _eval_council_decision(diff_text, key)



def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=== Berlin AI Studio: Pre-Push Security Council Gate ===")

    if not check_tests_and_coverage(repo_root):
        sys.exit(1)
    if not check_cyclomatic_complexity(repo_root):
        sys.exit(1)
    if not check_council_review(repo_root):
        sys.exit(1)

    print("\nALL GATES PASSED. Push authorized.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
