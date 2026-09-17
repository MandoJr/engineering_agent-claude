"""
Lightweight persistence for the engineering agent.

Uses plain JSON files under <project_root>/.engineering_agent/ so the
whole engineering memory is human-readable, diff-able, and greppable --
and so it imposes zero new infrastructure dependencies on JARVIS.

If JARVIS already has a lessons/history store (per the existing lessons
system), swap this module for an adapter around that store instead --
the rest of the package only depends on the methods defined here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonStore:
    """A tiny append/update JSON-lines-ish store, one JSON file per record,
    grouped into a directory per record type. Safe for single-process use."""

    def __init__(self, base_dir: Path, kind: str):
        self.dir = Path(base_dir) / kind
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, record_id: str) -> Path:
        return self.dir / f"{record_id}.json"

    def save(self, record_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            tmp = self._path(record_id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._path(record_id))

    def load(self, record_id: str) -> Optional[Dict[str, Any]]:
        p = self._path(record_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def delete(self, record_id: str) -> None:
        p = self._path(record_id)
        if p.exists():
            p.unlink()

    def list_all(self) -> List[Dict[str, Any]]:
        records = []
        for p in sorted(self.dir.glob("*.json"), key=lambda x: x.stat().st_mtime):
            try:
                records.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return records


class EngineeringStorage:
    """Facade grouping the stores the engineering agent needs."""

    def __init__(self, storage_dir: Path):
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs = JsonStore(storage_dir, "runs")
        self.proposals = JsonStore(storage_dir, "proposals")
        self.lessons = JsonStore(storage_dir, "lessons")
        self.self_improvement = JsonStore(storage_dir, "self_improvement")

    # -- convenience wrappers -------------------------------------------------

    def save_run(self, run) -> None:
        self.runs.save(run.run_id, run.to_dict())

    def save_proposal(self, proposal) -> None:
        self.proposals.save(proposal.proposal_id, proposal.to_dict())

    def save_lesson(self, lesson) -> None:
        self.lessons.save(lesson.lesson_id, lesson.to_dict())

    def all_runs(self) -> List[Dict[str, Any]]:
        return self.runs.list_all()

    def all_lessons(self) -> List[Dict[str, Any]]:
        return self.lessons.list_all()

    def all_proposals(self) -> List[Dict[str, Any]]:
        return self.proposals.list_all()
