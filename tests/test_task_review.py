import unittest

from engineering_agent.models import (
    EngineeringPlan,
    FileEdit,
    PatchResult,
    PlannedChange,
    TestResult as EngineeringTestResult,
)
from engineering_agent.task_graph import (
    EngineeringTask,
    TaskGraph,
    TaskStatus,
)
from engineering_agent.task_review import (
    IndependentTaskGraphReviewer,
)


class IndependentTaskGraphReviewTests(unittest.TestCase):
    def make_graph(self):
        return TaskGraph(
            [
                EngineeringTask(
                    task_id="A",
                    title="First",
                    objective="First task",
                    affected_files=["a.py"],
                    verification_commands=['python -c "print(1)"'],
                ),
                EngineeringTask(
                    task_id="B",
                    title="Second",
                    objective="Second task",
                    affected_files=["b.py"],
                    dependencies=["A"],
                    verification_commands=['python -c "print(1)"'],
                ),
            ]
        )

    def make_plan(self):
        return EngineeringPlan(
            plan_id="plan",
            files=["a.py", "b.py"],
            changes=[
                PlannedChange(file="a.py", description="first"),
                PlannedChange(
                    file="b.py",
                    description="second",
                    prerequisites=["a.py"],
                ),
            ],
            tests=['python -c "print(1)"'],
        )

    def mark_passed(self, graph):
        for task_id in ("A", "B"):
            graph.get(task_id).status = TaskStatus.PASSED
            graph.get(task_id).add_evidence(
                "VERIFICATION",
                "verification passed",
                {"passed_count": 1, "failed_count": 0},
            )

    def test_complete_graph_passes_independent_review(self):
        graph = self.make_graph()
        self.mark_passed(graph)

        review = IndependentTaskGraphReviewer().review(
            graph,
            self.make_plan(),
            [
                FileEdit(file="a.py", diff="@@ -1 +1 @@\n-old\n+new\n"),
                FileEdit(file="b.py", diff="@@ -1 +1 @@\n-old\n+new\n"),
            ],
            [EngineeringTestResult(command="python -c \"print(1)\"", passed=True)],
            [
                PatchResult(
                    change_id="a",
                    file="a.py",
                    operation="REPLACE_TEXT",
                    applied=True,
                ),
                PatchResult(
                    change_id="b",
                    file="b.py",
                    operation="REPLACE_TEXT",
                    applied=True,
                ),
            ],
        )

        self.assertTrue(review.passed)
        self.assertEqual(review.task_statuses["A"], "PASSED")
        self.assertEqual(review.task_statuses["B"], "PASSED")

    def test_blocked_task_fails_independent_review(self):
        graph = self.make_graph()
        graph.get("A").status = TaskStatus.FAILED
        graph.get("B").status = TaskStatus.BLOCKED

        review = IndependentTaskGraphReviewer().review(
            graph,
            self.make_plan(),
            [],
            [],
            [],
        )

        self.assertFalse(review.passed)
        self.assertTrue(
            any("FAILED" in finding for finding in review.findings)
        )
        self.assertTrue(
            any("BLOCKED" in finding for finding in review.findings)
        )
        self.assertEqual(review.blocked_by["B"], ["A"])

    def test_out_of_scope_git_diff_is_rejected(self):
        graph = self.make_graph()
        self.mark_passed(graph)

        review = IndependentTaskGraphReviewer().review(
            graph,
            self.make_plan(),
            [
                FileEdit(
                    file="a.py",
                    diff="@@\n-old\n+new\n",
                )
            ],
            [EngineeringTestResult(passed=True)],
            [
                PatchResult(
                    change_id="a",
                    file="a.py",
                    operation="REPLACE_TEXT",
                    applied=True,
                )
            ],
            git_diff="+++ b/a.py\n+++ b/secret.py\n",
        )

        self.assertFalse(review.passed)
        self.assertTrue(
            any("outside approved scope" in finding
                for finding in review.findings)
        )

    def test_passed_task_without_verification_evidence_is_rejected(self):
        graph = self.make_graph()
        graph.get("A").status = TaskStatus.PASSED
        graph.get("B").status = TaskStatus.PASSED

        review = IndependentTaskGraphReviewer().review(
            graph,
            self.make_plan(),
            [],
            [EngineeringTestResult(passed=True)],
            [],
        )

        self.assertFalse(review.passed)
        self.assertTrue(
            any("verification evidence" in finding
                for finding in review.findings)
        )


if __name__ == "__main__":
    unittest.main()

