"""CLI adapter for running official submission commands under the desktop runtime.

Examples:
    python -m experiments.bill_target74.desktop_friendly_runtime \
        --module experiments.bill_target74.submission_pipeline validate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "predict", "report", "check"))
    parser.add_argument("--month", action="append",
                        help="run a preflight on selected registered validation months")
    args = parser.parse_args(sys.argv[1:])
    if args.month:
        if args.command != "validate":
            parser.error("--month is only valid with validate")
        months = [str(pd.Period(value, freq="M")) for value in args.month]
        tag = "_".join(value.replace("-", "") for value in months)
        output_dir = Path(run.ROOT) / "outputs" / "submission" / "preflight" / tag
        result = run.validate(months=months, output_dir=output_dir)
    else:
        result = run.COMMANDS[args.command]()
    return 1 if result is False else 0


if __name__ == "__main__":
    # The desktop runtime wraps this module and treats SystemExit as a failure,
    # even for exit code 0. Report check failures with an exception instead.
    main()
