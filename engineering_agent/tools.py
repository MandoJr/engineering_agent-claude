"""
Controlled tool layer.

Every filesystem/git/shell action the agent can take goes through this
module so permissions, path safety, and forbidden-path checks are
enforced in one place instead of scattered across planner/implementer
code. Nothing here silently commits, pushes, or deletes -- those require
an explicit, separately-gated call (see git_utils.py and the COMMIT
permission level).
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .config import AgentConfig, SECRET_FILENAME_MARKERS, MAX_FILE_BYTES_FOR_CONTEXT


class Permission(str, Enum):
    READ_ONLY = "READ_ONLY"
    PROPOSE = "PROPOSE"     # may stage edits in-memory / to a diff, not on disk
    IMPLEMENT = "IMPLEMENT"  # may write to disk on approved changes
    COMMIT = "COMMIT"       # may run git commit (still requires explicit call)


class ToolError(RuntimeError):
    pass


class PermissionDenied(ToolError):
    pass


class PathNotAllowed(ToolError):
    pass


class ToolResult:
    def __init__(self, ok: bool, data=None, error: Optional[str] = None):
        self.ok = ok
        self.data = data
        self.error = error

    def __repr__(self):
        return f"ToolResult(ok={self.ok}, error={self.error!r})"


def _looks_like_secret(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in SECRET_FILENAME_MARKERS)


def _has_unquoted_shell_operator(command: str) -> bool:
    """Reject shell composition while permitting e.g. ``python -c 'a; b'``."""
    quote = None
    escaped = False
    index = 0
    operators = ("&&", "||", ";", ">", "<", "`", "$(")
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif char in ("'", '"'):
            quote = None if quote == char else (char if quote is None else quote)
        elif quote is None and any(command.startswith(operator, index) for operator in operators):
            return True
        index += 1
    return False


class ToolBox:
    """
    All tool calls are scoped to `config.project_root` and checked against
    `config.forbidden_paths`. Construct one ToolBox per engineering run
    with the minimum permission level that run actually needs.
    """

    def __init__(self, config: AgentConfig, permission: Permission = Permission.READ_ONLY):
        self.config = config
        self.permission = permission
        self.root = config.project_root

    # -- permission / path safety --------------------------------------------

    def _require(self, minimum: Permission):
        order = [Permission.READ_ONLY, Permission.PROPOSE,
                 Permission.IMPLEMENT, Permission.COMMIT]
        if order.index(self.permission) < order.index(minimum):
            raise PermissionDenied(
                f"Tool requires {minimum.value} permission, "
                f"but this ToolBox has {self.permission.value}"
            )

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathNotAllowed(f"'{relative_path}' escapes project root")

        relative = candidate.relative_to(self.root)
        rel_str = relative.as_posix()
        for forbidden in self.config.forbidden_paths:
            if forbidden in relative.parts or rel_str == forbidden or rel_str.startswith(forbidden.rstrip("/") + "/"):
                raise PathNotAllowed(f"'{relative_path}' is under forbidden path '{forbidden}'")

        if _looks_like_secret(candidate):
            raise PathNotAllowed(f"'{relative_path}' looks like a secret/credential file")

        return candidate

    # -- read tools (READ_ONLY) ----------------------------------------------

    def read_file(self, relative_path: str, max_bytes: int = MAX_FILE_BYTES_FOR_CONTEXT) -> ToolResult:
        self._require(Permission.READ_ONLY)
        try:
            path = self._resolve(relative_path)
            if not path.is_file():
                return ToolResult(False, error=f"Not a file: {relative_path}")
            data = path.read_bytes()
            truncated = len(data) > max_bytes
            text = data[:max_bytes].decode("utf-8", errors="replace")
            return ToolResult(True, data={"content": text, "truncated": truncated,
                                           "size_bytes": len(data)})
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    def list_files(self, relative_dir: str = ".", pattern: str = "*",
                    max_results: int = 500) -> ToolResult:
        self._require(Permission.READ_ONLY)
        try:
            path = self._resolve(relative_dir)
            if not path.is_dir():
                return ToolResult(False, error=f"Not a directory: {relative_dir}")
            results = []
            for p in path.rglob(pattern):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.root)
                rel_str = str(rel)
                if any(part in self.config.forbidden_paths for part in rel.parts):
                    continue
                results.append(rel_str)
                if len(results) >= max_results:
                    break
            return ToolResult(True, data=sorted(results))
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    def search_code(self, query: str, glob: str = "**/*.py",
                     max_results: int = 100, regex: bool = False) -> ToolResult:
        self._require(Permission.READ_ONLY)
        try:
            matches = []
            pattern = re.compile(query) if regex else None
            for p in self.root.rglob(glob):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.root)
                if any(part in self.config.forbidden_paths for part in rel.parts):
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), start=1):
                    hit = pattern.search(line) if regex else (query in line)
                    if hit:
                        matches.append({
                            "file": str(rel),
                            "line": lineno,
                            "text": line.strip()[:300],
                        })
                        if len(matches) >= max_results:
                            return ToolResult(True, data=matches)
            return ToolResult(True, data=matches)
        except (ToolError, re.error) as exc:
            return ToolResult(False, error=str(exc))

    def find_symbol(self, symbol: str, glob: str = "**/*.py") -> ToolResult:
        """Best-effort symbol finder: matches `def symbol`, `class symbol`,
        or `symbol =` at line start. Good enough for repo orientation;
        swap in a real language server / ctags integration for precision."""
        self._require(Permission.READ_ONLY)
        pattern = re.compile(
            rf"^\s*(def|class)\s+{re.escape(symbol)}\b|^\s*{re.escape(symbol)}\s*="
        )
        try:
            hits = []
            for p in self.root.rglob(glob):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.root)
                if any(part in self.config.forbidden_paths for part in rel.parts):
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if pattern.match(line):
                        hits.append({"file": str(rel), "line": lineno, "text": line.strip()})
            return ToolResult(True, data=hits)
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    # -- git read tools (READ_ONLY) ------------------------------------------

    def git_status(self) -> ToolResult:
        self._require(Permission.READ_ONLY)
        return self._run_git(["status", "--porcelain=v1", "-b"])

    def git_diff(self, relative_path: Optional[str] = None, staged: bool = False) -> ToolResult:
        self._require(Permission.READ_ONLY)
        args = ["diff"]
        if staged:
            args.append("--staged")
        if relative_path:
            try:
                self._resolve(relative_path)
            except ToolError as exc:
                return ToolResult(False, error=str(exc))
            args.append(relative_path)
        return self._run_git(args)

    def _run_git(self, args: List[str]) -> ToolResult:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=self.root,
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return ToolResult(False, error=proc.stderr.strip() or "git command failed")
            return ToolResult(True, data=proc.stdout)
        except (OSError, subprocess.SubprocessError) as exc:
            return ToolResult(False, error=str(exc))

    # -- write tools (IMPLEMENT) ---------------------------------------------

    def create_file(self, relative_path: str, content: str) -> ToolResult:
        self._require(Permission.IMPLEMENT)
        try:
            path = self._resolve(relative_path)
            if path.exists():
                return ToolResult(False, error=f"File already exists: {relative_path} "
                                                f"(use edit_file to modify it)")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            diff = "".join(difflib.unified_diff(
                [], content.splitlines(keepends=True),
                fromfile="/dev/null", tofile=relative_path,
            ))
            return ToolResult(True, data={"diff": diff, "bytes_after": len(content.encode("utf-8"))})
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    def edit_file(self, relative_path: str, new_content: str) -> ToolResult:
        """Targeted full-content replace with a real unified diff recorded.
        Callers (the implementer) should prefer minimal, localized changes
        by reading the file first and only altering what's necessary --
        this tool just guarantees the write is safe and diffed, it doesn't
        itself enforce minimality."""
        self._require(Permission.IMPLEMENT)
        try:
            path = self._resolve(relative_path)
            if not path.is_file():
                return ToolResult(False, error=f"File does not exist: {relative_path} "
                                                f"(use create_file to add it)")
            old_content = path.read_text(encoding="utf-8", errors="replace")
            if old_content == new_content:
                return ToolResult(True, data={"diff": "", "bytes_after": len(new_content.encode("utf-8")),
                                               "unchanged": True})
            diff = "".join(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=relative_path, tofile=relative_path,
            ))
            path.write_text(new_content, encoding="utf-8")
            return ToolResult(True, data={"diff": diff, "bytes_before": len(old_content.encode("utf-8")),
                                           "bytes_after": len(new_content.encode("utf-8"))})
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    def delete_file(self, relative_path: str) -> ToolResult:
        """Delete one approved file only when destructive operations are enabled.

        The operation is intentionally opt-in at configuration time; a model
        cannot enable it through a plan or a prompt.
        """
        self._require(Permission.IMPLEMENT)
        if not getattr(self.config, "allow_file_deletion", False):
            return ToolResult(False, error="File deletion is disabled by configuration")
        try:
            path = self._resolve(relative_path)
            if not path.is_file():
                return ToolResult(False, error=f"Not a file: {relative_path}")
            old = path.read_text(encoding="utf-8", errors="replace")
            path.unlink()
            diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), [], fromfile=relative_path, tofile="/dev/null"))
            return ToolResult(True, data={"diff": diff, "bytes_before": len(old.encode("utf-8")), "bytes_after": 0})
        except ToolError as exc:
            return ToolResult(False, error=str(exc))

    # -- test / command tools (READ_ONLY: running tests doesn't modify code) --

    def run_test(self, command: str, timeout: float = 300.0) -> ToolResult:
        self._require(Permission.READ_ONLY)
        return self._run_shell(command, timeout)

    def run_command(self, command: str, timeout: float = 120.0) -> ToolResult:
        """General command execution. Still READ_ONLY-gated (no destructive
        shell access is granted beyond running within the repo), and the
        caller is responsible for not requesting destructive commands --
        this is a development tool, not a sandboxed shell."""
        self._require(Permission.READ_ONLY)
        return self._run_shell(command, timeout)

    def _run_shell(self, command: str, timeout: float) -> ToolResult:
        normalized = command.strip().lower()
        if not any(normalized.startswith(prefix) for prefix in self.config.safe_command_prefixes):
            return ToolResult(False, error="Command blocked by verification allow-list")
        # Chaining/redirection is where a benign-looking test turns into a
        # side-effecting shell script. Verification commands must be one tool.
        if _has_unquoted_shell_operator(command):
            return ToolResult(False, error="Shell composition is blocked for verification commands")
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=timeout,
            )
            return ToolResult(
                proc.returncode == 0,
                data={
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout[-8000:],
                    "stderr": proc.stderr[-8000:],
                },
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, error=f"command timed out after {timeout}s")
        except OSError as exc:
            return ToolResult(False, error=str(exc))

    def analyze_error(self, stderr_text: str) -> ToolResult:
        """Lightweight structural parse of a traceback / error message so the
        debugger has something more than raw text to reason over. This is
        heuristic, not a replacement for the backend's own reasoning."""
        self._require(Permission.READ_ONLY)
        lines = stderr_text.strip().splitlines()
        file_refs = re.findall(r'File "([^"]+)", line (\d+)', stderr_text)
        exception_line = lines[-1] if lines else ""
        return ToolResult(True, data={
            "exception_summary": exception_line[:500],
            "file_references": [{"file": f, "line": int(l)} for f, l in file_refs],
            "raw_tail": "\n".join(lines[-30:]),
        })
