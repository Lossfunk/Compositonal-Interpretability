"""Make separate paper-style overall and transformation breakdown CSVs for C1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_EXPLICIT = Path("artifacts/cogitao_codex_c1_icl")
DEFAULT_IMPLICIT = Path("artifacts/cogitao_codex_c1_icl_implicit")
TASK_ORDER = {
    "CompGen-ID1": 0,
    "CompGen-ID2": 1,
    "CompGen-OOD1": 2,
    "CompGen-OOD2": 3,
}
RESULT_ORDER = {"function": 0, "composition": 1}
KEY_FIELDS = (
    "task", "distribution", "evaluation_set", "scope", "experiment",
    "result_type", "transformation",
)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_tables(
    explicit_dir: Path,
    implicit_dir: Path,
    overall_output: Path,
    breakdown_output: Path,
) -> tuple[int, int]:
    cells: dict[tuple[str, ...], dict[str, str]] = {}
    models: set[str] = set()
    for embedding, directory in (("Exp.", explicit_dir), ("Imp.", implicit_dir)):
        source = directory / "analysis.csv"
        if not source.is_file():
            continue
        with source.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                model = row["model"]
                models.add(model)
                key = (
                    row["condition"], row["distribution"], row["evaluation_set"],
                    row["scope"], row["experiment"], row["result_type"],
                    row["transformation"],
                )
                column = f"{model} {embedding}"
                samples = int(row["samples"])
                # Task totals appear only when all 50 C1 target calls are saved.
                if row["scope"] == "C1" and row["result_type"] == "overall":
                    score = f"{row['exact_grids']} / 50" if samples == 50 else ""
                else:
                    score = f"{row['exact_grids']} / {samples}" if samples else ""
                bucket = cells.setdefault(key, {})
                if column in bucket:
                    raise ValueError(f"Duplicate {embedding} result for {key}")
                bucket[column] = score

    columns = [f"{model} {embedding}" for model in sorted(models)
               for embedding in ("Exp.", "Imp.")]
    overall_keys = sorted(
        (key for key in cells if key[3] == "C1" and key[5] == "overall"),
        key=lambda key: (TASK_ORDER.get(key[0], 99), 0 if key[2] == "val" else 1),
    )
    overall_rows = [
        {
            "task": key[0], "distribution": key[1],
            "evaluation_set": key[2], **cells[key],
        }
        for key in overall_keys
    ]
    _write(overall_output,
           ["task", "distribution", "evaluation_set", *columns], overall_rows)

    breakdown_keys = sorted(
        (key for key in cells if key[3] == "C1" and key[5] in RESULT_ORDER),
        key=lambda key: (
            TASK_ORDER.get(key[0], 99), 0 if key[2] == "val" else 1,
            0 if key[3] == "C1" else 1,
            -1 if key[4] == "all" else int(key[4]),
            RESULT_ORDER[key[5]], key[6],
        ),
    )
    breakdown_rows = [
        {
            "task": key[0], "distribution": key[1],
            "evaluation_set": key[2], "result_type": key[5],
            "transformation": key[6], **cells[key],
        }
        for key in breakdown_keys
    ]
    _write(breakdown_output,
           ["task", "distribution", "evaluation_set", "result_type",
            "transformation", *columns], breakdown_rows)
    return len(overall_rows), len(breakdown_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-dir", type=Path, default=DEFAULT_EXPLICIT)
    parser.add_argument("--implicit-dir", type=Path, default=DEFAULT_IMPLICIT)
    parser.add_argument("--overall-output", type=Path,
                        default=DEFAULT_EXPLICIT / "table3_c1_overall.csv")
    parser.add_argument("--breakdown-output", type=Path,
                        default=DEFAULT_EXPLICIT / "table3_c1_breakdown.csv")
    args = parser.parse_args()
    overall_count, breakdown_count = build_tables(
        args.explicit_dir, args.implicit_dir,
        args.overall_output, args.breakdown_output,
    )
    print(f"Wrote {overall_count} overall rows to {args.overall_output}")
    print(f"Wrote {breakdown_count} breakdown rows to {args.breakdown_output}")


if __name__ == "__main__":
    main()
