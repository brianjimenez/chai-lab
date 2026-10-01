"""Merges the CSVs written by parallel `chai-lab score` runs into one CSV.

Usage: python merge_scores.py merged.csv part1.csv part2.csv ...

The columns are the union of the columns of the parts (per-chain columns depend
on each structure's chains); rows are sorted by aggregate_score, best first.
Fails if a structure appears more than once.
"""

import csv
import sys
from pathlib import Path


def merge(output: Path, parts: list[Path]) -> int:
    rows = []
    for part in parts:
        with open(part, newline="") as f:
            rows.extend(csv.DictReader(f))

    names = [row["structure"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Some structures are scored more than once")

    headers = list(dict.fromkeys(key for row in rows for key in row))
    rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    n = merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
    print(f"Wrote {n} structures to {sys.argv[1]}")
