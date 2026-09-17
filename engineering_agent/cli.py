"""
Development CLI for the engineering agent.

    python -m engineering_agent propose "Improve backend routing"
    python -m engineering_agent review <proposal_id>
    python -m engineering_agent approve <proposal_id>
    python -m engineering_agent reject <proposal_id> [--notes "..."]
    python -m engineering_agent implement <proposal_id>
    python -m engineering_agent test <run_id>
    python -m engineering_agent evaluate <run_id>
    python -m engineering_agent history [--limit N]
    python -m engineering_agent lessons [--limit N]
    python -m engineering_agent self-improve

By default the CLI operates on the current working directory as the
project root. Override with --root /path/to/jarvis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import ApprovalError, EngineeringOrchestrator


def _print(obj) -> None:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    elif isinstance(obj, list):
        obj = [o.to_dict() if hasattr(o, "to_dict") else o for o in obj]
    print(json.dumps(obj, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engineering_agent")
    parser.add_argument("--root", default=".", help="JARVIS project root (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose")
    p.add_argument("goal", help="Engineering goal, e.g. 'Improve backend routing'")
    p.add_argument("--profile", default=None, help="Force a task profile (ANALYZE, IMPLEMENT, ...)")

    p = sub.add_parser("review")
    p.add_argument("proposal_id")

    p = sub.add_parser("approve")
    p.add_argument("proposal_id")
    p.add_argument("--notes", default=None)

    p = sub.add_parser("reject")
    p.add_argument("proposal_id")
    p.add_argument("--notes", default=None)

    p = sub.add_parser("request-changes")
    p.add_argument("proposal_id")
    p.add_argument("notes")

    p = sub.add_parser("revise-plan")
    p.add_argument("proposal_id")
    p.add_argument("evidence", nargs="+", help="Observed failure/review evidence for the revised plan")

    p = sub.add_parser("implement")
    p.add_argument("proposal_id")

    p = sub.add_parser("test")
    p.add_argument("run_id")

    p = sub.add_parser("evaluate")
    p.add_argument("run_id")

    p = sub.add_parser("history")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("lessons")
    p.add_argument("--limit", type=int, default=50)

    sub.add_parser("self-improve")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    agent = EngineeringOrchestrator(project_root=str(Path(args.root).resolve()))

    try:
        if args.command == "propose":
            _print(agent.propose(args.goal, task_profile=args.profile))
        elif args.command == "review":
            _print(agent.review(args.proposal_id))
        elif args.command == "approve":
            _print(agent.approve(args.proposal_id, notes=args.notes))
        elif args.command == "reject":
            _print(agent.reject(args.proposal_id, notes=args.notes))
        elif args.command == "request-changes":
            _print(agent.request_changes(args.proposal_id, args.notes))
        elif args.command == "revise-plan":
            _print(agent.revise_plan(args.proposal_id, args.evidence))
        elif args.command == "implement":
            _print(agent.implement(args.proposal_id))
        elif args.command == "test":
            _print(agent.test(args.run_id))
        elif args.command == "evaluate":
            _print(agent.evaluate(args.run_id))
        elif args.command == "history":
            _print(agent.history(limit=args.limit))
        elif args.command == "lessons":
            _print(agent.lessons(limit=args.limit))
        elif args.command == "self-improve":
            result = agent.self_improve()
            _print(result if result else {"result": "No actionable weakness pattern detected yet."})
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except ApprovalError as exc:
        print(f"APPROVAL ERROR: {exc}", file=sys.stderr)
        return 3
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
