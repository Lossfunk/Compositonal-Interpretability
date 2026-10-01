"""Evaluate COGITAO C1 grids through the logged-in Codex CLI, without an API client."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SPLITS = ("val", "val_ood", "test", "test_ood")
MODELS = ("gpt-6-sol", "gpt-6-astra")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _split_path(root: Path, experiment: int, split: str) -> Path:
    return root / "exp_setting_1" / f"experiment_{experiment}" / f"{split}.parquet"


def _read_split(root: Path, experiment: int, split: str):
    path = _split_path(root, experiment, split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing C1 split: {path}")
    return pq.read_table(
        path, columns=["input", "output", "transformation_suite"], memory_map=True
    )


def _grid(table: Any, column: str, index: int) -> list[list[int]]:
    grid = table[column][index].as_py()
    if not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError(f"Invalid {column} grid at row {index}")
    width = len(grid[0])
    if any(len(row) != width or any(type(cell) is not int or not 0 <= cell <= 9 for cell in row)
           for row in grid):
        raise ValueError(f"Invalid C1 color or shape in {column} row {index}")
    return grid


def _render_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _demonstrations(root: Path, experiment: int, count: int, seed: int):
    table = _read_split(root, experiment, "train")
    suites = table["transformation_suite"].to_pylist()
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, suite in enumerate(suites):
        if len(suite) == 1:
            candidates[suite[0]].append(index)
    rng = random.Random(seed + experiment)
    examples: dict[str, list[dict[str, Any]]] = {}
    for function, indices in sorted(candidates.items()):
        if len(indices) < count:
            raise ValueError(f"Only {len(indices)} training examples for {function}")
        examples[function] = [
            {
                "source_index": index,
                "input": _grid(table, "input", index),
                "output": _grid(table, "output", index),
            }
            for index in rng.sample(indices, count)
        ]
    return examples


def _indices(table: Any, limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= len(table):
        return list(range(len(table)))
    # The parquet rows are grouped by task. Balance a small preview across suites.
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, suite in enumerate(table["transformation_suite"].to_pylist()):
        buckets[tuple(suite)].append(index)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: list[int] = []
    keys = sorted(buckets)
    while len(chosen) < limit:
        advanced = False
        for key in keys:
            if buckets[key] and len(chosen) < limit:
                chosen.append(buckets[key].pop())
                advanced = True
        if not advanced:
            break
    return sorted(chosen)


def _prompt(system_text: str, suite: list[str], input_grid: list[list[int]],
            examples: dict[str, list[dict[str, Any]]]) -> str:
    sections = [system_text.strip(), "", "LABELED TRAINING EXAMPLES:"]
    for function in dict.fromkeys(suite):
        if function not in examples:
            raise ValueError(f"No atomic training examples for {function}")
        for example in examples[function]:
            sections.extend([
                "", f"Transformation: {function}",
                "Input grid:", _render_grid(example["input"]),
                "Output grid:", _render_grid(example["output"]),
            ])
    sections.extend([
        "", "TEST INPUT:", _render_grid(input_grid),
        "", "Transformations to apply in order: " + " -> ".join(suite),
        "Return only the transformed output grid as a Python list of lists.",
    ])
    return "\n".join(sections) + "\n"


def _final_message(jsonl: str) -> str:
    messages = []
    for line in jsonl.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            messages.append(item.get("text", ""))
    if not messages:
        raise ValueError("Codex returned no final agent message")
    return messages[-1].strip()


def _codex_failure(stdout: str, stderr: str, returncode: int) -> str:
    messages = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"error", "turn.failed"}:
            error = event.get("error", event.get("message", ""))
            if isinstance(error, dict):
                error = error.get("message", error)
            if error:
                messages.append(str(error))
    if stderr.strip():
        messages.append(stderr.strip()[-2000:])
    if not messages and stdout.strip():
        messages.append(stdout.strip()[-2000:])
    detail = " | ".join(messages) if messages else "no diagnostic output"
    return f"Codex exited with code {returncode}: {detail}"


def _codex(prompt: str, model: str, workdir: Path, timeout: int) -> tuple[str, str, str]:
    command = [
        "codex", "exec", "--model", model, "--sandbox", "read-only",
        "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
        "--json", "-",
    ]
    environment = os.environ.copy()
    # Use the existing Codex CLI login, not an API key injected into the process.
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_API_KEY", None)
    try:
        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True,
            cwd=workdir, env=environment, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout", "", f"Codex exceeded {timeout} seconds"
    if completed.returncode:
        return "codex_error", "", _codex_failure(
            completed.stdout, completed.stderr, completed.returncode,
        )
    try:
        return "ok", _final_message(completed.stdout), ""
    except ValueError as exc:
        return "codex_error", "", str(exc)


def _parse_grid(text: str, target: list[list[int]]) -> list[list[int]]:
    response = text.strip()
    if not response.startswith("[") or not response.endswith("]"):
        raise ValueError("Response is not a bare Python list")
    try:
        grid = ast.literal_eval(response)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError("Response is not a Python list of lists") from exc
    if not isinstance(grid, list) or len(grid) != len(target):
        raise ValueError("Output grid has the wrong height")
    if any(not isinstance(row, list) or len(row) != len(target[0]) for row in grid):
        raise ValueError("Output grid has the wrong width")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("Output grid must contain integer colors 0-9")
    return grid


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[int(record["source_index"])] = record
    return records


def _summary(output_dir: Path) -> None:
    buckets: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in output_dir.glob("*/experiment_*/*.jsonl"):
        for record in _load_records(path).values():
            base = (record["model"], record["experiment"], record["split"])
            buckets[(*base, "all")].append(record)
            buckets[(*base, " -> ".join(record["suite"]))].append(record)
    summary_path = output_dir / "summary.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "model", "experiment", "split", "transformation_suite", "samples",
            "scored_samples", "valid_outputs", "invalid_outputs", "timeouts",
            "codex_errors", "exact_grids", "grid_accuracy_pct",
            "per_pixel_accuracy_pct",
        ])
        writer.writeheader()
        for key, records in sorted(buckets.items()):
            scored = [record for record in records if record["status"] in {"ok", "invalid_output"}]
            cells = sum(record["total_cells"] for record in scored)
            exact = sum(bool(record["exact_match"]) for record in scored)
            writer.writerow({
                "model": key[0], "experiment": key[1], "split": key[2],
                "transformation_suite": key[3], "samples": len(records),
                "scored_samples": len(scored),
                "valid_outputs": sum(record["status"] == "ok" for record in records),
                "invalid_outputs": sum(record["status"] == "invalid_output" for record in records),
                "timeouts": sum(record["status"] == "timeout" for record in records),
                "codex_errors": sum(record["status"] == "codex_error" for record in records),
                "exact_grids": exact,
                "grid_accuracy_pct": 100 * exact / len(scored) if scored else "",
                "per_pixel_accuracy_pct": 100 * sum(record["correct_cells"] for record in scored) / cells if cells else "",
            })
    temporary.replace(summary_path)
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--experiments", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["val", "test"])
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cogitao_codex_c1"))
    parser.add_argument("--iid-prompt", type=Path, default=PROMPT_DIR / "iid.txt")
    parser.add_argument("--ood-prompt", type=Path, default=PROMPT_DIR / "ood.txt")
    parser.add_argument("--examples-per-function", type=int, default=3)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--retry-failures", action="store_true",
        help="Retry previously saved timeout or CLI-error rows on resume.",
    )
    args = parser.parse_args()
    if args.examples_per_function <= 0 or args.timeout_seconds <= 0:
        parser.error("Example count and timeout must be positive")
    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        parser.error("--max-samples-per-split must be positive")
    if shutil.which("codex") is None:
        parser.error("Codex CLI is not on PATH; install it and run `codex login` first")
    prompt_texts = {}
    for domain, path in (("iid", args.iid_prompt), ("ood", args.ood_prompt)):
        if domain == "ood" and not any(split.endswith("_ood") for split in args.splits):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text or text.startswith("TODO:"):
            parser.error(f"{domain.upper()} prompt is still a placeholder: {path}")
        prompt_texts[domain] = text

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cogitao-codex-") as temporary_dir:
        isolated_dir = Path(temporary_dir)
        for experiment in args.experiments:
            examples = _demonstrations(args.data_root, experiment, args.examples_per_function, args.seed)
            example_ids = {name: [item["source_index"] for item in items] for name, items in examples.items()}
            for split in args.splits:
                table = _read_split(args.data_root, experiment, split)
                indices = _indices(table, args.max_samples_per_split, args.seed + experiment)
                domain = "ood" if split.endswith("_ood") else "iid"
                instructions = prompt_texts[domain]
                for model in args.models:
                    result_dir = args.output_dir / model / f"experiment_{experiment}"
                    result_dir.mkdir(parents=True, exist_ok=True)
                    result_path = result_dir / f"{split}.jsonl"
                    manifest_path = result_dir / f"{split}.manifest.json"
                    manifest = {
                        "model": model, "experiment": experiment, "split": split,
                        "source_indices": indices, "examples": example_ids,
                        "system_prompt_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
                    }
                    if manifest_path.exists():
                        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if previous != manifest:
                            raise ValueError(f"Existing result uses different prompts/examples: {manifest_path}")
                    elif result_path.exists():
                        raise ValueError(f"Result exists without a manifest: {result_path}")
                    else:
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    records = _load_records(result_path)
                    with result_path.open("a", encoding="utf-8") as file:
                        for index in indices:
                            if index in records and (
                                not args.retry_failures
                                or records[index]["status"] in {"ok", "invalid_output"}
                            ):
                                continue
                            suite = table["transformation_suite"][index].as_py()
                            input_grid = _grid(table, "input", index)
                            target = _grid(table, "output", index)
                            prompt = _prompt(instructions, suite, input_grid, examples)
                            started = time.monotonic()
                            status, response, error = _codex(
                                prompt, model, isolated_dir, args.timeout_seconds,
                            )
                            prediction = None
                            if status == "ok":
                                try:
                                    prediction = _parse_grid(response, target)
                                except ValueError as exc:
                                    status, error = "invalid_output", str(exc)
                            total_cells = sum(len(row) for row in target)
                            correct_cells = sum(
                                predicted == expected
                                for pred_row, target_row in zip(prediction, target)
                                for predicted, expected in zip(pred_row, target_row)
                            ) if prediction is not None else 0
                            record = {
                                "model": model, "experiment": experiment, "split": split,
                                "source_index": index, "suite": suite, "status": status,
                                "response": response, "error": error,
                                "prediction": prediction, "target": target,
                                "exact_match": prediction == target if prediction is not None else False,
                                "correct_cells": correct_cells, "total_cells": total_cells,
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                            }
                            file.write(json.dumps(record) + "\n")
                            file.flush()
                            records[index] = record
                            print(f"{model} C1/E{experiment} {split} row {index}: {status}", flush=True)
                            if status == "codex_error":
                                _summary(args.output_dir)
                                raise RuntimeError(f"Codex failed on {model} {split} row {index}: {error}")
                            # A slow row is saved as an unscored timeout; continue
                            # through the remaining samples.
    _summary(args.output_dir)


if __name__ == "__main__":
    main()
