"""
engineering_agent
==================

Self-improving software-engineering agent for JARVIS.

This package is designed to be dropped into an existing JARVIS codebase
with minimal architectural disruption. It does NOT replace JARVIS's
agency/model-worker, routing, health, or lessons systems -- it extends
them through small adapter interfaces (see `integration.py`).

Public entry point:

    from engineering_agent import EngineeringOrchestrator

    agent = EngineeringOrchestrator(project_root="/path/to/jarvis")
    proposal = agent.propose("Improve backend routing")
    ...
    agent.approve(proposal.proposal_id)
    result = agent.implement(proposal.proposal_id)
    evaluation = agent.evaluate(result.run_id)
    agent.self_improve()
"""

from .orchestrator import EngineeringOrchestrator
from .models import (
    EngineeringGoal,
    EngineeringPlan,
    EngineeringProposal,
    EngineeringRun,
    EngineeringLesson,
    ApprovalStatus,
    RiskLevel,
    HealthState,
    TaskProfileName,
)

from .execution_state import (
    ExecutionStateError,
    InvalidTaskTransitionError,
    StateTransition,
    TaskExecutionStateMachine,
)
from .task_decomposer import (
    TaskDecomposer,
    TaskDecompositionError,
)
from .task_recovery import (
    TaskRecoveryCoordinator,
    TaskRecoveryOutcome,
)
from .task_verification import (
    TaskVerificationEngine,
    TaskVerificationOutcome,
)
from .task_execution import (
    TaskExecutionEngine,
    TaskExecutionOutcome,
)
from .task_graph import (
    DependencyCycleError,
    DuplicateTaskError,
    EngineeringTask,
    TaskEvidence,
    TaskGraph,
    TaskGraphError,
    TaskStatus,
    UnknownTaskError,
)

__all__ = [
    "EngineeringOrchestrator",
    "EngineeringGoal",
    "EngineeringPlan",
    "EngineeringProposal",
    "EngineeringRun",
    "EngineeringLesson",
    "ApprovalStatus",
    "RiskLevel",
    "HealthState",
    "TaskProfileName",
    "TaskExecutionStateMachine",
"StateTransition",
"ExecutionStateError",
"InvalidTaskTransitionError",
"TaskDecomposer",
"TaskDecompositionError",
"TaskExecutionEngine",
"TaskExecutionOutcome",
    "TaskVerificationEngine",
    "TaskRecoveryCoordinator",
    "TaskRecoveryOutcome",
    "TaskVerificationOutcome",
"EngineeringTask",
    "TaskEvidence",
    "TaskGraph",
    "TaskStatus",
    "TaskGraphError",
    "DuplicateTaskError",
    "UnknownTaskError",
    "DependencyCycleError",
]

__version__ = "0.1.0"
