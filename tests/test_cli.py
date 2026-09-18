import unittest

from engineering_agent.cli import build_parser


class CliTests(unittest.TestCase):
    def test_resume_command_exists(self):
        parser = build_parser()

        args = parser.parse_args(["resume", "run-123"])

        self.assertEqual(args.command, "resume")
        self.assertEqual(args.run_id, "run-123")


if __name__ == "__main__":
    unittest.main()
