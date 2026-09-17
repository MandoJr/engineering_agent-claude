"""
Configuration for the engineering agent.

These constants mirror the tuning knobs already present in JARVIS's
routing / lessons systems (agency_model_worker) so the engineering
agent's usage of those systems stays consistent rather than introducing
a second, conflicting set of thresholds. If JARVIS's real router/lessons
modules are wired in via integration.py, THOSE modules' own constants
are authoritative -- these are only used by the standalone fallbacks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Routing / learning constants (mirrors existing JARVIS router concepts)
# ---------------------------------------------------------------------------

EXPLORATION_INTERVAL = 10
LEARNING_MIN_EXECUTIONS = 5
BASELINE_MIN_EXECUTIONS = 5
LEARNED_ROUTING_BONUS_CAP = 0.20
LESSONS_MIN_CONFIDENCE = 0.75

# ---------------------------------------------------------------------------
# Safety / bounding constants
# ---------------------------------------------------------------------------

MAX_RECOVERY_ATTEMPTS = 3
MAX_FILES_PER_CHANGE = 25          # guardrail against runaway edits
MAX_FILE_BYTES_FOR_CONTEXT = 200_000  # cap on how much of one file we read into a prompt

# Paths the agent will never touch, even with approval, relative to project root.
DEFAULT_FORBIDDEN_PATHS = [
    ".git",
    ".env",
    "secrets",
    "credentials",
    ".ssh",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
]

# Filename fragments that always block EDIT/CREATE/COMMIT tools, regardless
# of forbidden-path config, because they typically hold secrets.
SECRET_FILENAME_MARKERS = [
    "id_rsa", ".pem", ".key", "credentials.json", ".env",
    "secret", "token", "password",
]


@dataclass
class TaskProfile:
    """Defines expectations and requirements for one kind of engineering task."""
    name: str
    reasoning_requirement: str          # e.g. "deep_reasoning", "code_generation"
    required_backend_capabilities: List[str]
    requires_tests: bool
    requires_evaluation: bool
    default_risk: str                   # RiskLevel value
    requires_user_approval: bool
    recovery_strategy: str              # "retry_with_analysis" | "abort" | "escalate"
    max_recovery_attempts: int = MAX_RECOVERY_ATTEMPTS


TASK_PROFILES: Dict[str, TaskProfile] = {
    "ANALYZE": TaskProfile(
        name="ANALYZE",
        reasoning_requirement="deep_reasoning",
        required_backend_capabilities=["repository_analysis"],
        requires_tests=False,
        requires_evaluation=False,
        default_risk="LOW",
        requires_user_approval=False,
        recovery_strategy="abort",
    ),
    "PLAN": TaskProfile(
        name="PLAN",
        reasoning_requirement="deep_reasoning",
        required_backend_capabilities=["deep_reasoning", "code_generation"],
        requires_tests=False,
        requires_evaluation=False,
        default_risk="LOW",
        requires_user_approval=False,
        recovery_strategy="abort",
    ),
    "IMPLEMENT": TaskProfile(
        name="IMPLEMENT",
        reasoning_requirement="code_generation",
        required_backend_capabilities=["code_generation"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="MEDIUM",
        requires_user_approval=True,
        recovery_strategy="retry_with_analysis",
    ),
    "DEBUG": TaskProfile(
        name="DEBUG",
        reasoning_requirement="debugging",
        required_backend_capabilities=["debugging", "code_generation"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="MEDIUM",
        requires_user_approval=True,
        recovery_strategy="retry_with_analysis",
    ),
    "REVIEW": TaskProfile(
        name="REVIEW",
        reasoning_requirement="code_review",
        required_backend_capabilities=["code_review"],
        requires_tests=False,
        requires_evaluation=False,
        default_risk="LOW",
        requires_user_approval=False,
        recovery_strategy="abort",
    ),
    "TEST": TaskProfile(
        name="TEST",
        reasoning_requirement="code_generation",
        required_backend_capabilities=["code_generation"],
        requires_tests=True,
        requires_evaluation=False,
        default_risk="LOW",
        requires_user_approval=False,
        recovery_strategy="retry_with_analysis",
    ),
    "REFACTOR": TaskProfile(
        name="REFACTOR",
        reasoning_requirement="code_generation",
        required_backend_capabilities=["code_generation", "repository_analysis"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="MEDIUM",
        requires_user_approval=True,
        recovery_strategy="retry_with_analysis",
    ),
    "OPTIMIZE": TaskProfile(
        name="OPTIMIZE",
        reasoning_requirement="deep_reasoning",
        required_backend_capabilities=["code_generation", "repository_analysis"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="MEDIUM",
        requires_user_approval=True,
        recovery_strategy="retry_with_analysis",
    ),
    "ARCHITECTURE": TaskProfile(
        name="ARCHITECTURE",
        reasoning_requirement="deep_reasoning",
        required_backend_capabilities=["deep_reasoning", "repository_analysis"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="HIGH",
        requires_user_approval=True,
        recovery_strategy="escalate",
    ),
    "SELF_IMPROVE": TaskProfile(
        name="SELF_IMPROVE",
        reasoning_requirement="deep_reasoning",
        required_backend_capabilities=["deep_reasoning", "code_generation"],
        requires_tests=True,
        requires_evaluation=True,
        default_risk="HIGH",
        requires_user_approval=True,
        recovery_strategy="escalate",
    ),
}


@dataclass
class AgentConfig:
    project_root: Path
    ollama_endpoint: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen3-coder:30b"
    storage_dir: Path = None
    forbidden_paths: List[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_PATHS))
    max_recovery_attempts: int = MAX_RECOVERY_ATTEMPTS
    max_files_per_change: int = MAX_FILES_PER_CHANGE

    def __post_init__(self):
        self.project_root = Path(self.project_root).resolve()
        if self.storage_dir is None:
            self.storage_dir = self.project_root / ".engineering_agent"
        else:
            self.storage_dir = Path(self.storage_dir)
