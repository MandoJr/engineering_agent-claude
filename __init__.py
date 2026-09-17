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
]

__version__ = "0.1.0"
