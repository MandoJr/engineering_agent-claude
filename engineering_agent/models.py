"""
Core data models for the engineering agent.

Everything that flows through the engineering loop (goals, plans,
proposals, runs, lessons) is a plain, JSON-serializable dataclass so it
can be persisted, inspected, and rendered by the CLI or by JARVIS itself
without extra glue code.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILING = "FAILING"
    UNKNOWN = "UNKNOWN"


class TaskProfileName(str, Enum):
    ANALYZE = "ANALYZE"
    PLAN = "PLAN"
    IMPLEMENT = "IMPLEMENT"
    DEBUG = "DEBUG"
    REVIEW = "REVIEW"
    TEST = "TEST"
    REFACTOR = "REFACTOR"
    OPTIMIZE = "OPTIMIZE"
    ARCHITECTURE = "ARCHITECTURE"
    SELF_IMPROVE = "SELF_IMPROVE"


class RunStatus(str, Enum):
    CREATED = "CREATED"
    ANALYZING = "ANALYZING"
    PLANNED = "PLANNED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    REJECTED = "REJECTED"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    RECOVERING = "RECOVERING"
    EVALUATING = "EVALUATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


def _serialize(obj: Any) -> Any:
    """Recursively turn dataclasses/enums into JSON-safe plain data."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


class _Base:
    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)


# ---------------------------------------------------------------------------
# Goal / classification
# ---------------------------------------------------------------------------

@dataclass
class EngineeringGoal(_Base):
    goal_id: str = field(default_factory=lambda: _new_id("goal"))
    description: str = ""
    task_profile: str = TaskProfileName.ANALYZE.value
    created_at: float = field(default_factory=_now)
    requested_by: str = "user"
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Repository understanding
# ---------------------------------------------------------------------------

@dataclass
class RepositoryContext(_Base):
    root: str = ""
    affected_systems: List[str] = field(default_factory=list)
    candidate_files: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    related_config: List[str] = field(default_factory=list)
    symbols: Dict[str, List[str]] = field(default_factory=dict)   # file -> symbol names
    git_status_summary: str = ""
    module_graph: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class PlannedChange(_Base):
    file: str = ""
    description: str = ""
    change_type: str = "MODIFY"   # MODIFY | CREATE | DELETE
    objective: str = ""
    prerequisites: List[str] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    risk: str = RiskLevel.LOW.value


@dataclass
class EngineeringPlan(_Base):
    plan_id: str = field(default_factory=lambda: _new_id("plan"))
    goal_id: str = ""
    affected_systems: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    changes: List[PlannedChange] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)
    expected_result: str = ""
    risk_level: str = RiskLevel.LOW.value
    created_at: float = field(default_factory=_now)
    raw_backend_output: Optional[str] = None
    revision: int = 0
    assumptions: List[str] = field(default_factory=list)
    verification_requirements: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Proposal (the approval-gated artifact)
# ---------------------------------------------------------------------------

@dataclass
class EngineeringProposal(_Base):
    proposal_id: str = field(default_factory=lambda: _new_id("prop"))
    goal: EngineeringGoal = None
    plan: EngineeringPlan = None
    affected_files: List[str] = field(default_factory=list)
    risk: str = RiskLevel.LOW.value
    expected_improvement: str = ""
    required_approval: str = "USER"
    status: str = ApprovalStatus.WAITING_FOR_APPROVAL.value
    created_at: float = field(default_factory=_now)
    decided_at: Optional[float] = None
    decision_notes: Optional[str] = None
    is_self_improvement: bool = False


# ---------------------------------------------------------------------------
# Diffs / implementation
# ---------------------------------------------------------------------------

class ChangeOperation(str, Enum):
    CREATE_FILE = "CREATE_FILE"
    REPLACE_TEXT = "REPLACE_TEXT"
    INSERT_BEFORE = "INSERT_BEFORE"
    INSERT_AFTER = "INSERT_AFTER"
    REPLACE_SYMBOL = "REPLACE_SYMBOL"
    DELETE_REGION = "DELETE_REGION"
    DELETE_FILE = "DELETE_FILE"
    MOVE_FILE = "MOVE_FILE"


@dataclass
class StructuredChange(_Base):
    """A small, verifiable intent; never an implicit whole-file overwrite."""
    change_id: str = field(default_factory=lambda: _new_id("chg"))
    file: str = ""
    operation: str = ChangeOperation.REPLACE_TEXT.value
    target_symbol: Optional[str] = None
    expected_content: Optional[str] = None
    content: str = ""
    target_file: Optional[str] = None
    reason: str = ""
    task_id: Optional[str] = None
    risk: str = RiskLevel.LOW.value
    validation_requirements: List[str] = field(default_factory=list)


@dataclass
class ChangeSet(_Base):
    changes: List[StructuredChange] = field(default_factory=list)
    task_id: Optional[str] = None
    rationale: str = ""


@dataclass
class PatchResult(_Base):
    change_id: str = ""
    file: str = ""
    operation: str = ""
    applied: bool = False
    classification: str = ""
    message: str = ""
    diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    rollback_performed: bool = False

@dataclass
class FileEdit(_Base):
    file: str = ""
    change_type: str = "MODIFY"
    diff: str = ""            # unified diff as applied
    bytes_before: int = 0
    bytes_after: int = 0


@dataclass
class ImplementationResult(_Base):
    run_id: str = ""
    edits: List[FileEdit] = field(default_factory=list)
    backend_used: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    patch_results: List[PatchResult] = field(default_factory=list)
    rollback_performed: bool = False


# ---------------------------------------------------------------------------
# Testing / recovery
# ---------------------------------------------------------------------------

@dataclass
class TestResult(_Base):
    command: str = ""
    passed: bool = False
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0
    classification: str = ""


@dataclass
class RecoveryAttempt(_Base):
    attempt_number: int = 0
    root_cause: str = ""
    recovery_plan: str = ""
    edits: List[FileEdit] = field(default_factory=list)
    test_result: Optional[TestResult] = None
    succeeded: bool = False
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation / benchmark
# ---------------------------------------------------------------------------

@dataclass
class HealthSnapshot(_Base):
    system: str = ""
    state: str = HealthState.UNKNOWN.value
    detail: str = ""
    taken_at: float = field(default_factory=_now)


@dataclass
class BenchmarkResult(_Base):
    name: str = ""
    baseline_value: Optional[float] = None
    post_change_value: Optional[float] = None
    unit: str = ""
    improved: Optional[bool] = None
    notes: str = ""


@dataclass
class EvaluationResult(_Base):
    run_id: str = ""
    code_works: bool = False
    improved_jarvis: Optional[bool] = None   # None = insufficient evidence
    evidence: List[str] = field(default_factory=list)
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    health_before: List[HealthSnapshot] = field(default_factory=list)
    health_after: List[HealthSnapshot] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=_now)

@dataclass
class ReviewResult(_Base):
    passed: bool = False
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lessons (engineering memory)
# ---------------------------------------------------------------------------

@dataclass
class EngineeringLesson(_Base):
    lesson_id: str = field(default_factory=lambda: _new_id("lesson"))
    task: str = ""
    context: str = ""
    attempt: str = ""
    result: str = ""
    failure: Optional[str] = None
    cause: Optional[str] = None
    solution: Optional[str] = None
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=_now)
    source_run_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engineering run (the full history record)
# ---------------------------------------------------------------------------

@dataclass
class EngineeringRun(_Base):
    run_id: str = field(default_factory=lambda: _new_id("run"))
    goal: EngineeringGoal = None
    task_profile: str = TaskProfileName.ANALYZE.value
    repository_context: Optional[RepositoryContext] = None
    plan: Optional[EngineeringPlan] = None
    proposal_id: Optional[str] = None
    approval: str = ApprovalStatus.NOT_REQUIRED.value
    backend: Optional[str] = None
    implementation: Optional[ImplementationResult] = None
    tests: List[TestResult] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    recovery_attempts: List[RecoveryAttempt] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    review: Optional[ReviewResult] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    task_graph: Optional[Dict[str, Any]] = None
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    current_task_id: Optional[str] = None
    status: str = RunStatus.CREATED.value
    final_result: Optional[str] = None
    lessons_created: List[str] = field(default_factory=list)   # lesson_ids
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def touch(self):
        self.updated_at = _now()
