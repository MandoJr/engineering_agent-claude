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
import os
from typing import List

from .models import TestResult
from .tools import ToolBox


class TestEngineer:
    def __init__(self, tools: ToolBox):
        self.tools = tools

    def run_tests(self, test_specs: List[str]) -> List[TestResult]:
        if not test_specs:
            test_specs = self.discover_tests()
        if not test_specs:
            return [TestResult(command="(none discovered)", passed=False,
                               stderr_tail="No test command was planned or discovered.", classification="NO_TESTS")]

        results = []
        for spec in test_specs[:self.tools.config.max_test_commands]:
            command = spec if self._looks_like_command(spec) else f"python -m pytest {spec} -q"
            results.append(self._run_one(command))
        return results

    def discover_tests(self) -> List[str]:
        """Discover a safe project-native test command without guessing success."""
        root = self.tools.root
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or any(root.glob("test*.py")) or (root / "tests").is_dir():
            return ["python -m pytest -q"]
        if (root / "package.json").is_file():
            return ["npm test"]
        if (root / "Cargo.toml").is_file():
            return ["cargo test"]
        if (root / "go.mod").is_file():
            return ["go test ./..."]
        return []

    def _looks_like_command(self, spec: str) -> bool:
        return " " in spec.strip() or spec.strip().startswith(("python", "pytest", "npm", "make"))

    def _run_one(self, command: str) -> TestResult:
        # Plans are often authored on Unix where `python3` is conventional;
        # use the Windows launcher name when executing on Windows.
        if os.name == "nt" and command.startswith("python3 "):
            command = "python " + command[len("python3 "):]
        start = time.time()
        result = self.tools.run_test(command)
        duration = time.time() - start
        if result.ok:
            data = result.data
            return TestResult(
                command=command, passed=True, exit_code=data["exit_code"],
                stdout_tail=data["stdout"][-2000:], stderr_tail=data["stderr"][-2000:],
                duration_s=duration,
                classification="PASS",
            )
        data = result.data or {}
        return TestResult(
            command=command, passed=False, exit_code=data.get("exit_code"),
            stdout_tail=data.get("stdout", "")[-2000:],
            stderr_tail=(data.get("stderr", "") or result.error or "")[-2000:],
            duration_s=duration,
            classification=self._classify_failure(data.get("stderr", "") or result.error or ""),
        )

    def _classify_failure(self, output: str) -> str:
        lowered = output.lower()
        if "timed out" in lowered: return "TIMEOUT"
        if "syntaxerror" in lowered or "importerror" in lowered or "modulenotfounderror" in lowered: return "ENVIRONMENT_OR_IMPORT"
        if "assert" in lowered or "failed" in lowered: return "ASSERTION_FAILURE"
        if "not found" in lowered or "command blocked" in lowered: return "COMMAND_CONFIGURATION"
        return "UNKNOWN_FAILURE"
