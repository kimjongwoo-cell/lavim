# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# python 0912/completed_indices.py RUN_ROOT RUN_PREFIX DATASET

from __future__ import annotations

import argparse
import json
from pathlib import Path


def completed_indices(
    run_root: Path,
    *,
    run_prefix: str,
    dataset: str,
) -> tuple[int, ...]:
    """Return completed IDs whose recorded slide belongs to exactly one dataset."""
    completed: set[int] = set()
    expected_prefix = f"{dataset}__"
    for result_path in run_root.glob(f"{run_prefix}*/*/result.json"):
        try:
            payload = json.loads(result_path.read_text())
            if str(payload["slide_id"]).startswith(expected_prefix):
                completed.add(int(payload["dataset_index"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return tuple(sorted(completed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("run_prefix")
    parser.add_argument("dataset")
    args = parser.parse_args()
    for index in completed_indices(
        args.run_root,
        run_prefix=args.run_prefix,
        dataset=args.dataset,
    ):
        print(index)


if __name__ == "__main__":
    main()

