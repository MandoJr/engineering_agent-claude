"""
Safe git workflow helpers.

Hard rule enforced here: `commit()` and `push()` both require an explicit
`approved=True` argument passed by the orchestrator, which only does so
after checking the proposal's ApprovalStatus is APPROVED. There is no
code path in this module that commits or pushes as a side effect of
anything else (branch creation, staging, diffing are all safe/reversible
and don't require approval).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class GitError(RuntimeError):
    pass


class ApprovalRequired(GitError):
    pass


@dataclass
class GitStatus:
    branch: str
    clean: bool
    raw: str


class GitWorkflow:
    def __init__(self, project_root: Path):
        self.root = Path(project_root)
        if not (self.root / ".git").exists():
            raise GitError(f"{self.root} is not a git repository")

    def _run(self, args: List[str], timeout: float = 30.0) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=self.root,
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise GitError(proc.stderr.strip() or f"git {' '.join(args)} failed")
        return proc.stdout

    def status(self) -> GitStatus:
        raw = self._run(["status", "--porcelain=v1", "-b"])
        lines = raw.splitlines()
        branch_line = lines[0] if lines else ""
        branch = branch_line.replace("## ", "").split("...")[0].strip() or "HEAD"
        clean = len(lines) <= 1
        return GitStatus(branch=branch, clean=clean, raw=raw)

    def diff(self, staged: bool = False, path: Optional[str] = None) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            args.append(path)
        return self._run(args)

    def current_branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"]).strip()

    def create_branch(self, name: str, checkout: bool = True) -> None:
        """Creating and switching branches is safe/reversible -- no approval gate."""
        if checkout:
            self._run(["checkout", "-b", name])
        else:
            self._run(["branch", name])

    def stage(self, paths: List[str]) -> None:
        """Staging is reversible (git reset) -- no approval gate, but it's
        the last step before the approval-gated commit()."""
        if not paths:
            raise GitError("No paths given to stage")
        self._run(["add", *paths])

    def unstage_all(self) -> None:
        self._run(["reset"])

    def commit(self, message: str, approved: bool) -> str:
        if not approved:
            raise ApprovalRequired(
                "Refusing to commit: caller did not pass approved=True. "
                "Commits require explicit, checked user approval upstream."
            )
        return self._run(["commit", "-m", message])

    def push(self, remote: str = "origin", branch: Optional[str] = None,
              approved: bool = False) -> str:
        if not approved:
            raise ApprovalRequired(
                "Refusing to push: caller did not pass approved=True. "
                "Pushes require explicit, checked user approval upstream."
            )
        branch = branch or self.current_branch()
        return self._run(["push", remote, branch])

    def log(self, max_count: int = 10) -> str:
        return self._run(["log", f"-{max_count}", "--oneline"])
