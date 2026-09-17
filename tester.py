"""
Test Engineer.

Runs the tests named in a plan (file paths or bare commands) and returns
structured TestResults. Test file paths are run through pytest if
available; anything that isn't a discoverable test path is treated as a
literal shell command (so a plan can specify "pytest tests/ -k routing"
directly).
"""

from __future__ import annotations

import time
from typing import List

from .models import TestResult
from .tools import ToolBox


class TestEngineer:
    def __init__(self, tools: ToolBox):
        self.tools = tools

    def run_tests(self, test_specs: List[str]) -> List[TestResult]:
        if not test_specs:
            return [TestResult(
                command="(none specified)", passed=False,
                stdout_tail="", stderr_tail="No tests were specified for this change.",
            )]

        results = []
        for spec in test_specs:
            command = spec if self._looks_like_command(spec) else f"python -m pytest {spec} -q"
            results.append(self._run_one(command))
        return results

    def _looks_like_command(self, spec: str) -> bool:
        return " " in spec.strip() or spec.strip().startswith(("python", "pytest", "npm", "make"))

    def _run_one(self, command: str) -> TestResult:
        start = time.time()
        result = self.tools.run_test(command)
        duration = time.time() - start
        if result.ok:
            data = result.data
            return TestResult(
                command=command, passed=True, exit_code=data["exit_code"],
                stdout_tail=data["stdout"][-2000:], stderr_tail=data["stderr"][-2000:],
                duration_s=duration,
            )
        data = result.data or {}
        return TestResult(
            command=command, passed=False, exit_code=data.get("exit_code"),
            stdout_tail=data.get("stdout", "")[-2000:],
            stderr_tail=(data.get("stderr", "") or result.error or "")[-2000:],
            duration_s=duration,
        )
