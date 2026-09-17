"""
Repository Analyst.

Builds a structured RepositoryContext for a goal before any planning
happens, so the planner/implementer never work from a bare prompt alone.
Combines heuristic static analysis (imports, keyword search, test/config
discovery) with the backend's own reasoning over that structured
evidence, rather than asking the backend to "figure out the codebase"
from nothing.
"""

from __future__ import annotations

import re
from typing import List

from .backend import ModelBackend
from .models import RepositoryContext
from .tools import ToolBox

TEST_DIR_HINTS = ("test", "tests", "spec", "specs")
CONFIG_FILE_HINTS = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env.example")

# Keyword -> JARVIS subsystem, used to seed affected_systems from a plain
# English goal before the backend gets involved. Extend this as JARVIS's
# real subsystem names solidify.
SYSTEM_KEYWORDS = {
    "rout": "routing",
    "backend": "backend",
    "health": "health",
    "lesson": "lessons",
    "benchmark": "benchmark",
    "debug": "debugging",
    "test": "testing",
    "approval": "approval",
    "engineering": "engineering_agent",
    "agency": "agency",
    "model worker": "agency_model_worker",
}


class RepositoryAnalyst:
    def __init__(self, tools: ToolBox, backend: ModelBackend):
        self.tools = tools
        self.backend = backend

    def analyze(self, goal_description: str, max_candidate_files: int = 20) -> RepositoryContext:
        affected_systems = self._guess_systems(goal_description)

        candidate_files: List[str] = []
        seen = set()
        for keyword, system in SYSTEM_KEYWORDS.items():
            if system not in affected_systems:
                continue
            result = self.tools.search_code(keyword.split()[0], glob="**/*.py", max_results=15)
            if result.ok:
                for hit in result.data:
                    f = hit["file"]
                    if f not in seen:
                        seen.add(f)
                        candidate_files.append(f)
            if len(candidate_files) >= max_candidate_files:
                break

        related_tests = self._find_related_tests(candidate_files)
        related_config = self._find_related_config()

        symbols = {}
        for f in candidate_files[:10]:
            read = self.tools.read_file(f, max_bytes=20_000)
            if read.ok:
                symbols[f] = self._extract_top_level_symbols(read.data["content"])

        status = self.tools.git_status()
        git_summary = status.data if status.ok else (status.error or "unavailable")

        return RepositoryContext(
            root=str(self.tools.root),
            affected_systems=affected_systems,
            candidate_files=candidate_files[:max_candidate_files],
            related_tests=related_tests,
            related_config=related_config,
            symbols=symbols,
            git_status_summary=git_summary,
        )

    # -- heuristics -----------------------------------------------------------

    def _guess_systems(self, goal_description: str) -> List[str]:
        lowered = goal_description.lower()
        found = []
        for keyword, system in SYSTEM_KEYWORDS.items():
            if keyword in lowered and system not in found:
                found.append(system)
        return found or ["unclassified"]

    def _find_related_tests(self, candidate_files: List[str]) -> List[str]:
        tests = []
        for f in candidate_files:
            stem = f.split("/")[-1].replace(".py", "")
            result = self.tools.search_code(stem, glob="**/test_*.py", max_results=5)
            if result.ok:
                tests.extend(hit["file"] for hit in result.data)
            result = self.tools.search_code(stem, glob="**/*_test.py", max_results=5)
            if result.ok:
                tests.extend(hit["file"] for hit in result.data)
        return sorted(set(tests))

    def _find_related_config(self) -> List[str]:
        configs = []
        for ext in CONFIG_FILE_HINTS:
            result = self.tools.list_files(".", pattern=f"*{ext}", max_results=20)
            if result.ok:
                configs.extend(result.data)
        return sorted(set(configs))[:20]

    def _extract_top_level_symbols(self, source: str) -> List[str]:
        pattern = re.compile(r"^(?:def|class)\s+(\w+)")
        return [m.group(1) for line in source.splitlines()
                if (m := pattern.match(line))]
