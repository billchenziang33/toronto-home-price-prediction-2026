from experiments.bill_target74 import submission_pipeline


def test_successful_submission_command_returns_without_system_exit(monkeypatch):
    monkeypatch.setattr(submission_pipeline.sys, "argv", ["submission_pipeline.py", "report"])
    monkeypatch.setitem(submission_pipeline.run.COMMANDS, "report", lambda: None)

    assert submission_pipeline.main() == 0
