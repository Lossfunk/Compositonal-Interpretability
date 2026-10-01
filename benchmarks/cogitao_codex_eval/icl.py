"""Five-example C1 ICL benchmark through parallel Codex CLI workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.cogitao_codex_eval.run import (
    MODELS,
    PROMPT_DIR,
    SPLITS,
    _codex,
    _grid,
    _load_records,
    _parse_grid,
    _read_split,
    _render_grid,
)

CONDITIONS = {
    "CompGen-ID1": ("iid", 2, 3),
    "CompGen-ID2": ("iid", 0, 5),
    "CompGen-OOD1": ("ood", 2, 3),
    "CompGen-OOD2": ("ood", 0, 5),
}
DEFAULT_CONDITIONS = {
    "iid": ("CompGen-ID1", "CompGen-ID2"),
    "ood": ("CompGen-OOD1", "CompGen-OOD2"),
}
PROTOCOL_VERSION = 1
EMBEDDINGS = ("explicit", "implicit")
DEFAULT_OUTPUT_DIR = Path("artifacts/cogitao_codex_c1_icl")
DEFAULT_IMPLICIT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.with_name(DEFAULT_OUTPUT_DIR.name + "_implicit")


@dataclass(frozen=True)
class Example:
    source_index: int
    suite: tuple[str, ...]
    input_grid: list[list[int]]
    output_grid: list[list[int]]


@dataclass(frozen=True)
class Job:
    model: str
    experiment: int
    split: str
    condition: str
    source_index: int
    target_suite: tuple[str, ...]
    target_grid: list[list[int]]
    prompt: str
    embedding: str


def _suite_indices(table: Any) -> dict[tuple[str, ...], list[int]]:
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, suite in enumerate(table["transformation_suite"].to_pylist()):
        buckets[tuple(suite)].append(index)
    return dict(buckets)


def _choose_id_suite(
    train_suites: dict[tuple[str, ...], list[int]],
    ood_suite: tuple[str, ...],
) -> tuple[str, ...]:
    seen_pairs = [suite for suite in train_suites if len(suite) == 2]
    reverse = tuple(reversed(ood_suite))
    if reverse in train_suites and reverse != ood_suite:
        return reverse
    heterogeneous = [suite for suite in seen_pairs if suite[0] != suite[1]]
    for suite in heterogeneous:
        if suite[0] == ood_suite[0]:
            return suite
    for suite in heterogeneous:
        if set(suite) & set(ood_suite):
            return suite
    raise ValueError(f"No suitable seen composed suite for {ood_suite}")


def _id_overrides(path: Path | None) -> dict[int, tuple[str, ...]]:
    if path is None:
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("ID suite mapping must be a JSON object")
    overrides: dict[int, tuple[str, ...]] = {}
    for key, value in source.items():
        experiment = int(key)
        if experiment not in range(1, 6) or not isinstance(value, list) or len(value) != 2:
            raise ValueError("ID suite mapping needs experiment keys 1-5 and two-function lists")
        overrides[experiment] = tuple(str(part) for part in value)
    return overrides


def _target_indices(indices: list[int], count: int, seed: int) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"Requested {count} targets but suite has only {len(indices)} rows")
    return sorted(random.Random(seed).sample(indices, count))


def _pick_context(
    table: Any,
    train_suites: dict[tuple[str, ...], list[int]],
    target_suite: tuple[str, ...],
    condition: str,
    seed: int,
) -> list[Example]:
    _, singles_needed, composed_needed = CONDITIONS[condition]
    rng = random.Random(seed)
    relevant = set(target_suite)
    single_suites = [suite for suite in train_suites if len(suite) == 1 and suite[0] in relevant]
    composed_suites = [
        suite for suite in train_suites
        if len(suite) == 2 and any(function in relevant for function in suite)
    ]
    if singles_needed and not single_suites:
        raise ValueError(f"No atomic training context for {target_suite}")
    if composed_needed and not composed_suites:
        raise ValueError(f"No composed training context for {target_suite}")
    # Rotate through distinct suites before reusing one. This keeps composed
    # examples varied and ensures both atomic functions appear when possible.
    rng.shuffle(single_suites)
    rng.shuffle(composed_suites)
    chosen_suites: list[tuple[str, ...]] = []
    if singles_needed and len(single_suites) >= 2:
        chosen_suites.extend(single_suites[:2])
    while len(chosen_suites) < singles_needed:
        chosen_suites.append(single_suites[len(chosen_suites) % len(single_suites)])
    for index in range(composed_needed):
        chosen_suites.append(composed_suites[index % len(composed_suites)])
    used: set[int] = set()
    examples = []
    for suite in chosen_suites:
        candidates = train_suites[suite]
        source_index = rng.choice(candidates)
        while source_index in used:
            source_index = rng.choice(candidates)
        used.add(source_index)
        examples.append(Example(
            source_index=source_index,
            suite=suite,
            input_grid=_grid(table, "input", source_index),
            output_grid=_grid(table, "output", source_index),
        ))
    if len(examples) != 5:
        raise AssertionError("Every ICL prompt must contain exactly five examples")
    return examples


def _code_map(train_suites: dict[tuple[str, ...], list[int]],
              target_suite: tuple[str, ...]) -> dict[str, str]:
    """Use stable, opaque names for every atomic transformation in an experiment."""
    functions = sorted({function for suite in train_suites for function in suite})
    if not set(target_suite).issubset(functions):
        raise ValueError(f"Implicit target contains a function absent from training: {target_suite}")
    return {function: f"t{index}" for index, function in enumerate(functions, 1)}


def _prompt(base_prompt: str, examples: list[Example], suite: tuple[str, ...],
            input_grid: list[list[int]],
            code_map: dict[str, str] | None = None) -> str:
    def labels(parts: tuple[str, ...]) -> str:
        if code_map is not None:
            return json.dumps([code_map[part] for part in parts])
        return " -> ".join(parts)

    sections = [base_prompt.strip(), "", "LABELED EXAMPLES (5):"]
    for number, example in enumerate(examples, 1):
        sections.extend([
            "", f"Example {number}",
            "Transformations: " + labels(example.suite),
            "Input grid:", _render_grid(example.input_grid),
            "Output grid:", _render_grid(example.output_grid),
        ])
    sections.extend([
        "", "TEST INPUT:", _render_grid(input_grid),
        "", "Transformations to apply in order: " + labels(suite),
        "Return only the transformed output grid as a Python list of lists.",
    ])
    return "\n".join(sections) + "\n"


def _evaluate(job: Job, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    # One empty working directory per CLI process avoids worker interference.
    with tempfile.TemporaryDirectory(prefix="cogitao-c1-icl-") as temporary:
        status, response, error = _codex(job.prompt, job.model, Path(temporary), timeout)
    prediction = None
    if status == "ok":
        try:
            prediction = _parse_grid(response, job.target_grid)
        except ValueError as exc:
            status, error = "invalid_output", str(exc)
    total_cells = sum(len(row) for row in job.target_grid)
    correct_cells = (
        sum(
            actual == expected
            for actual_row, expected_row in zip(prediction, job.target_grid)
            for actual, expected in zip(actual_row, expected_row)
        )
        if prediction is not None else 0
    )
    return {
        "model": job.model,
        "experiment": job.experiment,
        "split": job.split,
        "condition": job.condition,
        "embedding": job.embedding,
        "source_index": job.source_index,
        "suite": list(job.target_suite),
        "status": status,
        "response": response,
        "error": error,
        "prediction": prediction,
        "target": job.target_grid,
        "exact_match": prediction == job.target_grid if prediction is not None else False,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _summary(output_dir: Path) -> None:
    """Write overall, constituent-function, and ordered-composition scores."""
    buckets: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in output_dir.glob("*/experiment_*/*/*.jsonl"):
        for record in _load_records(path).values():
            model = str(record["model"])
            experiment = str(record["experiment"])
            split = str(record["split"])
            condition = str(record["condition"])
            suite = tuple(record["suite"])
            composition = " -> ".join(suite)
            for experiment_scope in ("all", experiment):
                prefix = (model, experiment_scope, split, condition)
                buckets[(*prefix, "overall", "all")].append(record)
                buckets[(*prefix, "composition", composition)].append(record)
                # Repeated operators in a composition count once per target.
                for function in set(suite):
                    buckets[(*prefix, "function", function)].append(record)
            buckets[(model, "all", "all", "all", "overall", "all")].append(record)

    fields = [
        "model", "experiment", "split", "condition", "granularity", "group",
        "samples", "scored_samples", "valid_outputs", "invalid_outputs",
        "timeouts", "codex_errors", "exact_grids", "grid_accuracy_pct",
        "per_pixel_accuracy_pct",
    ]
    rows = []
    for key, records in sorted(buckets.items()):
        scored = [record for record in records if record["status"] in {"ok", "invalid_output"}]
        cells = sum(record["total_cells"] for record in scored)
        exact = sum(bool(record["exact_match"]) for record in scored)
        rows.append({
            "model": key[0], "experiment": key[1], "split": key[2],
            "condition": key[3], "granularity": key[4], "group": key[5],
            "samples": len(records), "scored_samples": len(scored),
            "valid_outputs": sum(record["status"] == "ok" for record in records),
            "invalid_outputs": sum(record["status"] == "invalid_output" for record in records),
            "timeouts": sum(record["status"] == "timeout" for record in records),
            "codex_errors": sum(record["status"] == "codex_error" for record in records),
            "exact_grids": exact,
            "grid_accuracy_pct": 100 * exact / len(scored) if scored else "",
            "per_pixel_accuracy_pct": (
                100 * sum(record["correct_cells"] for record in scored) / cells
                if cells else ""
            ),
        })

    for name, selected in (
        ("summary.csv", rows),
        ("overall.csv", [row for row in rows if row["granularity"] == "overall"]),
        ("by_function.csv", [row for row in rows if row["granularity"] == "function"]),
        ("by_composition.csv", [row for row in rows if row["granularity"] == "composition"]),
    ):
        output = output_dir / name
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        temporary.replace(output)

    # A single tidy table for comparing IID and OOD across validation and test.
    # The grand total above spans both domains, so omit it from this table.
    analysis_fields = [
        "model", "distribution", "evaluation_set", "source_split", "scope",
        "experiment", "condition", "result_type", "transformation",
        "samples", "scored_samples", "valid_outputs", "invalid_outputs",
        "timeouts", "codex_errors", "exact_grids", "grid_accuracy_pct",
        "per_pixel_accuracy_pct",
    ]
    analysis_rows = []
    for row in rows:
        if row["split"] == "all":
            continue
        analysis_rows.append({
            "model": row["model"],
            "distribution": "OOD" if row["split"].endswith("_ood") else "IID",
            "evaluation_set": "test" if row["split"].startswith("test") else "val",
            "source_split": row["split"],
            "scope": "C1" if row["experiment"] == "all" else "experiment",
            "experiment": row["experiment"],
            "condition": row["condition"],
            "result_type": row["granularity"],
            "transformation": row["group"],
            **{field: row[field] for field in fields[6:]},
        })
    analysis_rows.sort(key=lambda row: (
        row["model"], row["distribution"], row["evaluation_set"],
        row["scope"], row["experiment"], row["condition"],
        row["result_type"], row["transformation"],
    ))
    analysis_path = output_dir / "analysis.csv"
    temporary = analysis_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=analysis_fields)
        writer.writeheader()
        writer.writerows(analysis_rows)
    temporary.replace(analysis_path)
    print(f"Wrote overall, function, composition, and combined analysis CSVs under {output_dir}", flush=True)
    if output_dir.resolve() in {DEFAULT_OUTPUT_DIR.resolve(), DEFAULT_IMPLICIT_OUTPUT_DIR.resolve()}:
        from benchmarks.cogitao_codex_eval.table3_c1 import build_tables
        build_tables(DEFAULT_OUTPUT_DIR, DEFAULT_IMPLICIT_OUTPUT_DIR,
                     DEFAULT_OUTPUT_DIR / "table3_c1_overall.csv",
                     DEFAULT_OUTPUT_DIR / "table3_c1_breakdown.csv")


def _run_group(
    *,
    output_dir: Path,
    model: str,
    experiment: int,
    split: str,
    condition: str,
    base_prompt: str,
    target_suite: tuple[str, ...],
    target_indices: list[int],
    target_table: Any,
    training_table: Any,
    training_suites: dict[tuple[str, ...], list[int]],
    seed: int,
    workers: int,
    timeout: int,
    retry_failures: bool,
    embedding: str,
    code_map: dict[str, str] | None,
) -> None:
    result_dir = output_dir / model / f"experiment_{experiment}" / split
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{condition}.jsonl"
    manifest_path = result_dir / f"{condition}.manifest.json"
    contexts: dict[int, list[Example]] = {}
    for index in target_indices:
        contexts[index] = _pick_context(
            training_table, training_suites, target_suite, condition,
            seed + experiment * 100_000 + index,
        )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model": model, "experiment": experiment, "split": split,
        "condition": condition, "target_suite": list(target_suite),
        "target_indices": target_indices,
        "context": {
            str(index): [
                {"source_index": example.source_index, "suite": list(example.suite)}
                for example in examples
            ]
            for index, examples in contexts.items()
        },
        "base_prompt_sha256": hashlib.sha256(base_prompt.encode()).hexdigest(),
    }
    # Keep existing explicit manifests compatible with prior runs.
    if embedding == "implicit":
        manifest["embedding"] = embedding
        manifest["task_code_map"] = code_map
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise ValueError(f"Existing ICL results use a different prompt or sample plan: {manifest_path}")
    elif result_path.exists():
        raise ValueError(f"ICL result exists without manifest: {result_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    previous = _load_records(result_path)
    jobs = []
    for index in target_indices:
        if index in previous and (
            not retry_failures or previous[index]["status"] in {"ok", "invalid_output"}
        ):
            continue
        actual_suite = tuple(target_table["transformation_suite"][index].as_py())
        if actual_suite != target_suite:
            raise ValueError(f"Target suite changed at {split} row {index}")
        jobs.append(Job(
            model=model, experiment=experiment, split=split,
            condition=condition, source_index=index, target_suite=target_suite,
            target_grid=_grid(target_table, "output", index),
            prompt=_prompt(
                base_prompt, contexts[index], target_suite,
                _grid(target_table, "input", index), code_map,
            ),
            embedding=embedding,
        ))
    if not jobs:
        print(f"{model} E{experiment} {split} {condition}: already complete", flush=True)
        return
    print(
        f"{model} E{experiment} {split} {condition}: {len(jobs)} prompts "
        f"with {min(workers, len(jobs))} workers", flush=True,
    )
    with result_path.open("a", encoding="utf-8") as file:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_evaluate, job, timeout): job for job in jobs}
            for future in as_completed(futures):
                record = future.result()
                file.write(json.dumps(record) + "\n")
                file.flush()
                print(
                    f"{record['model']} E{experiment} {split} {condition} "
                    f"row {record['source_index']}: {record['status']}", flush=True,
                )
    _summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--embedding", choices=EMBEDDINGS, default="explicit",
                        help="Use transformation names or opaque task codes in prompts")
    parser.add_argument("--experiments", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["val", "test"])
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS))
    parser.add_argument("--targets-per-experiment", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--id-target-suites", type=Path, help="Optional JSON mapping experiment 1-5 to a seen two-function suite")
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--iid-prompt", type=Path)
    parser.add_argument("--ood-prompt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.iid_prompt is None:
        args.iid_prompt = PROMPT_DIR / ("iid_implicit.txt" if args.embedding == "implicit" else "iid.txt")
    if args.ood_prompt is None:
        args.ood_prompt = PROMPT_DIR / ("ood_implicit.txt" if args.embedding == "implicit" else "ood.txt")
    if args.output_dir is None:
        args.output_dir = (DEFAULT_IMPLICIT_OUTPUT_DIR if args.embedding == "implicit"
                           else DEFAULT_OUTPUT_DIR)
    if args.workers <= 0 or args.targets_per_experiment <= 0 or args.timeout_seconds <= 0:
        parser.error("Workers, targets, and timeout must be positive")
    if shutil.which("codex") is None:
        parser.error("Codex CLI is not on PATH; install it and run `codex login` first")
    overrides = _id_overrides(args.id_target_suites)
    prompts = {}
    for domain, path in (("iid", args.iid_prompt), ("ood", args.ood_prompt)):
        if domain == "ood" and not any(split.endswith("_ood") for split in args.splits):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text or text.startswith("TODO:"):
            parser.error(f"{domain.upper()} base prompt is still a placeholder: {path}")
        prompts[domain] = text
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        train = _read_split(args.data_root, experiment, "train")
        train_suites = _suite_indices(train)
        ood_reference = _read_split(args.data_root, experiment, "val_ood")
        ood_suites = _suite_indices(ood_reference)
        if len(ood_suites) != 1:
            raise ValueError(f"Expected one held-out C1 suite in experiment {experiment}")
        ood_suite = next(iter(ood_suites))
        if len(ood_suite) != 2 or ood_suite in train_suites:
            raise ValueError(f"Expected an unseen two-function suite in experiment {experiment}")
        code_map = _code_map(train_suites, ood_suite) if args.embedding == "implicit" else None
        id_suite = overrides.get(experiment) or _choose_id_suite(train_suites, ood_suite)
        if len(id_suite) != 2 or id_suite not in train_suites:
            raise ValueError(f"ID target suite must be a seen two-function suite: {id_suite}")
        for split in args.splits:
            domain = "ood" if split.endswith("_ood") else "iid"
            conditions = args.conditions or DEFAULT_CONDITIONS[domain]
            if any(CONDITIONS[condition][0] != domain for condition in conditions):
                parser.error(f"Conditions {conditions} do not match {split}")
            target_suite = ood_suite if domain == "ood" else id_suite
            table = _read_split(args.data_root, experiment, split)
            buckets = _suite_indices(table)
            if target_suite not in buckets:
                raise ValueError(f"Target suite {target_suite} absent from {split} experiment {experiment}")
            # The same ten targets are used for both ICL conditions and models.
            indices = _target_indices(
                buckets[target_suite], args.targets_per_experiment,
                args.seed + experiment * 10_000 + SPLITS.index(split),
            )
            for condition in conditions:
                for model in args.models:
                    _run_group(
                        output_dir=args.output_dir, model=model,
                        experiment=experiment, split=split, condition=condition,
                        base_prompt=prompts[domain], target_suite=target_suite,
                        target_indices=indices, target_table=table,
                        training_table=train, training_suites=train_suites,
                        seed=args.seed, workers=args.workers,
                        timeout=args.timeout_seconds, retry_failures=args.retry_failures,
                        embedding=args.embedding, code_map=code_map,
                    )
    _summary(args.output_dir)


if __name__ == "__main__":
    main()
