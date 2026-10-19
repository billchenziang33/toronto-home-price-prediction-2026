"""Check the two submission files before you hand them in.

    python check_submission.py            (or: python run.py check)

Each file must have exactly two columns, ml_num,predicted_price, and one row for
every home of its own test file: no home missing, none twice, no other ids.
Every predicted price must be a positive, finite number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = ["off_market", "list_aware"]


def check_file(path, test_ids):
    """Return a list of problems with one submission file (an empty list means it is fine)."""
    try:
        submission = pd.read_csv(path, dtype={"ml_num": str})
    except Exception as error:
        return [f"cannot read the file: {error}"]
    if list(submission.columns) != ["ml_num", "predicted_price"]:
        return [f"the columns must be exactly ml_num,predicted_price, not {','.join(map(str, submission.columns))}"]

    problems = []
    ids = submission["ml_num"]
    expected = set(test_ids)
    missing = expected - set(ids)
    extra = set(ids) - expected
    if ids.duplicated().any():
        problems.append(f"{ids.duplicated().sum()} ml_num values appear more than once")
    if missing:
        problems.append(f"{len(missing)} homes of the test file are missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} ml_num values are not in this test file, e.g. {sorted(map(str, extra))[:3]}")

    price = pd.to_numeric(submission["predicted_price"], errors="coerce")
    if price.isna().any():
        problems.append(f"{price.isna().sum()} predicted prices are empty or not numbers")
    if np.isinf(price).any():
        problems.append(f"{np.isinf(price).sum()} predicted prices are infinite")
    if (price <= 0).any():
        problems.append(f"{(price <= 0).sum()} predicted prices are zero or negative")
    return problems


def check_all(folder="submissions"):
    """Check submission_off_market.csv and submission_list_aware.csv. Return True if both are fine."""
    import data
    all_fine = True
    for track in TRACKS:
        path = Path(folder) / f"submission_{track}.csv"
        if not path.exists():
            problems = ["the file does not exist yet: run `python run.py predict` first"]
        else:
            problems = check_file(path, data.load(track)["ml_num"])
        print(f"{path.name}: " + ("OK" if not problems else "PROBLEMS"))
        for problem in problems:
            print("   -", problem)
        all_fine = all_fine and not problems
    return all_fine


if __name__ == "__main__":
    folder = Path(__file__).parent / "submissions"
    sys.exit(0 if check_all(folder) else 1)
