"""Run frozen-latent group tests after successful rotation training completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import ATOMIC_ANGLES, ALL_PAIRS
from benchmarks.cogitao_vit_cross_attention.degree_rotation_model import DegreeRotationViT
from benchmarks.cogitao_vit_cross_attention.object_degree_rotation_data import ObjectAnchorDegreeRotationDataset
from benchmarks.cogitao_vit_cross_attention.rotation_compositions import predict
from benchmarks.cogitao_vit_cross_attention.rotation_group_probes import (
    GROUP_ANGLES, apply_operator, fit_operator, fit_projector, group_laws,
    latent_metrics, numeric_summary, probe_fit_metrics, project,
)
from benchmarks.cogitao_vit_cross_attention.run_rotations import write_json


def process_running(pid, marker):
    path = Path(f"/proc/{pid}")
    try:
        return marker in (path / "cmdline").read_text() and (path / "stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_for_training(output, destination, wait, poll_seconds=15):
    while True:
        status = json.loads((output / "status.json").read_text())
        if status["status"] in ("failed", "superseded"):
            raise RuntimeError(f"Training ended with status {status['status']}; no group tests run.")
        pid_path = output / "pid.txt"
        training_alive = pid_path.exists() and process_running(int(pid_path.read_text()), "run_degree_rotations")
        if status["status"] == "complete" and not training_alive:
            report = json.loads((output / "report.json").read_text())
            if report["status"] != "complete":
                raise RuntimeError("Training report is incomplete.")
            return report
        if not wait:
            raise RuntimeError("Group tests require successful completed training; use --wait to queue them.")
        if not training_alive:
            raise RuntimeError("Training process exited without a completed report; no group tests run.")
        write_json(destination / "status.json", {
            "status": "waiting_for_training", "training_status": status["status"],
            "training_epoch": status.get("current_epoch"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(poll_seconds)


def effect_programs(seen_pairs):
    """Use trained tokenizations for known effects; mark new single angles OOD."""
    programs, kinds = {}, {}
    for angle in GROUP_ANGLES:
        if angle in ATOMIC_ANGLES:
            programs[angle], kinds[angle] = (angle,), "atomic_trained"
        else:
            candidates = [tuple(pair) for pair in seen_pairs if sum(pair) == angle]
            if candidates:
                programs[angle], kinds[angle] = min(candidates), "composition_effect_trained"
            else:
                programs[angle], kinds[angle] = (angle,), "untrained_numeric_angle"
    return programs, kinds


@torch.inference_mode()
def extract_features(model, sources, program, batch_size=128, pool_grid=6):
    """Frozen final image tokens; spatial pooling retains coarse location.

    Numeric angle features extend the shared sin/cos projection for diagnostic
    identity/inverse queries. They do not alter or retrain checkpoint weights.
    Empty program means no active task tokens, not a trained identity token.
    """
    device = next(model.parameters()).device
    values = []
    model.eval()
    for start in range(0, len(sources), batch_size):
        grids = sources[start:start + batch_size].long().to(device)
        images = F.one_hot(grids, 10).permute(0, 3, 1, 2).float()
        tokens = torch.zeros(len(grids), 2, dtype=torch.long, device=device)
        tokens[:, :len(program)] = 1
        degrees = torch.zeros_like(tokens, dtype=torch.float32)
        if program:
            degrees[:, :len(program)] = torch.tensor(program, dtype=torch.float32, device=device)
        radians = degrees * (torch.pi / 180)
        angle_features = torch.stack((radians.sin(), radians.cos()), -1)
        angle_features[tokens == 0] = 0

        def numeric_embedding(module, inputs, output):
            return module.projection(angle_features)

        hook = model.task_embedding.register_forward_hook(numeric_embedding)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                output = predict(model, images, tokens)
            hidden = output["image_embeddings"].float()
            rows, cols = model.patch_grid
            hidden = hidden.transpose(1, 2).reshape(len(grids), model.hidden_dim, rows, cols)
            if pool_grid > min(rows, cols):
                raise ValueError("Spatial pooling grid exceeds the model's patch grid.")
            values.append(F.adaptive_avg_pool2d(hidden, (pool_grid, pool_grid)).flatten(1).cpu())
        finally:
            hook.remove()
    return torch.cat(values)


def extract_programs(model, dataset, programs, args, cache=None):
    cache = {} if cache is None else cache
    for program in dict.fromkeys(tuple(program) for program in programs):
        if program not in cache:
            cache[program] = extract_features(model, dataset.sources, program, args.batch_size, args.pool_grid)
    return cache


def latent_program_tests(dataset, features, projector, operators, canonical):
    rows, regrouped = [], []
    source = project(features[()], projector)
    indices = {program: i for i, program in enumerate(dataset.programs)}
    for a, b in ALL_PAIRS:
        program = (a, b)
        index = indices[program]
        raster_exact = dataset.raster_valid[:, index] & dataset.raster_references[:, index].eq(
            dataset.targets[:, (a + b) // 15 - 1]
        ).flatten(1).all(1)
        stable = dataset.identities_preserved[:, index]
        subsets = {"all": torch.ones(len(source), dtype=torch.bool),
                   "raster_exact_and_identity_preserved": raster_exact & stable}
        for subset, mask in subsets.items():
            if not bool(mask.any()):
                continue
            direct = features[program][mask]
            reversed_state = features[(b, a)][mask]
            equivalent = features[canonical[a + b]][mask]
            projected_direct = project(direct, projector)
            predicted_composition = apply_operator(operators[b] @ operators[a], source[mask])
            rows.append({
                "a": a, "b": b, "total": a + b, "subset": subset,
                "pair_split": "seen_pairs" if program in dataset.seen_pairs else "held_out_pairs",
                "samples": int(mask.sum()), "canonical_program": list(canonical[a + b]),
                **{f"raw_reverse_{key}": value for key, value in latent_metrics(direct, reversed_state).items()},
                **{f"raw_equivalent_effect_{key}": value for key, value in latent_metrics(direct, equivalent).items()},
                **{f"probe_composition_vs_pair_{key}": value for key, value in latent_metrics(predicted_composition, projected_direct).items()},
                **{f"probe_sum_vs_pair_{key}": value for key, value in latent_metrics(
                    apply_operator(operators[a + b], source[mask]), projected_direct
                ).items()},
            })
    for a, b, c in itertools.product(ATOMIC_ANGLES, repeat=3):
        if a + b > 90 or b + c > 90:
            continue
        left, right = (a + b, c), (a, b + c)
        regrouped.append({
            "a": a, "b": b, "c": c, "total": a + b + c,
            "left_program": list(left), "right_program": list(right),
            "left_pair_seen": left in dataset.seen_pairs,
            "right_pair_seen": right in dataset.seen_pairs,
            "samples": len(source),
            **latent_metrics(features[left], features[right]),
        })
    return rows, regrouped


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def publish_metrics(output, summary):
    """Resume the existing approved W&B run, uploading numeric scores only."""
    import wandb
    root = Path(__file__).resolve().parents[2]
    approval = json.loads((root / "runs/wandb_logging_authorization.json").read_text())
    if not approval.get("approved"):
        raise RuntimeError("W&B metrics logging is not authorized.")
    metadata = json.loads((output / "wandb.json").read_text())
    if metadata["project"] != approval["project"]:
        raise RuntimeError("W&B project does not match the logging authorization.")
    relay_pid = output / "wandb_bridge_pid.txt"
    while relay_pid.exists() and process_running(int(relay_pid.read_text()), "rotation_wandb"):
        time.sleep(5)
    run = wandb.init(
        entity=metadata["entity"], project=metadata["project"], id=metadata["id"],
        resume="must", dir=str(output), save_code=False,
        settings=wandb.Settings(disable_git=True, disable_code=True,
                                disable_job_creation=True, x_disable_stats=True),
    )

    def walk(value, prefix):
        for key, item in value.items():
            name = f"{prefix}/{key}"
            if isinstance(item, dict):
                walk(item, name)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                run.summary[name] = item

    walk(summary, "group_theory")
    url = run.url
    run.finish()
    return url


def evaluate(output, destination, training, args, initialization_control=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "data_manifest.json").read_text())
    datasets = {}
    for split in ("train", "val", "test"):
        specification = config["data"][split]
        if not specification["class_path"].endswith(".ObjectAnchorDegreeRotationDataset"):
            raise ValueError("These tests require the object-anchor degree-rotation experiment.")
        options = dict(specification["init_args"])
        if split == "train":
            options["max_sources"] = min(args.fit_sources, options.get("max_sources", args.fit_sources))
        if split != "train":
            # Include held-out pairs only in post-training diagnostic evaluation.
            options["program_selection"] = "all"
        datasets[split] = ObjectAnchorDegreeRotationDataset(options)
        saved_hashes = set(manifest[split]["input_sha256"])
        if not set(datasets[split].digests) <= saved_hashes:
            raise RuntimeError(f"{split} sources differ from the completed training run.")
        if split != "train" and set(datasets[split].digests) != saved_hashes:
            raise RuntimeError(f"{split} must evaluate all original held-out sources.")
        if datasets[split].rotation_convention != manifest[split]["rotation_convention"]:
            raise RuntimeError("Rotation convention changed since training.")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(datasets[left].digests) & set(datasets[right].digests):
            raise RuntimeError("Probe fitting and evaluation sources overlap.")
    checkpoint = Path(training["best_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if initialization_control:
        checkpoint = output / "initial_weights.pt"
        model = DegreeRotationViT(config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    else:
        model = DegreeRotationViT.load_from_checkpoint(checkpoint, map_location="cpu")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval().requires_grad_(False)
    canonical, kinds = effect_programs(datasets["train"].seen_pairs)
    extrapolated = [angle for angle, kind in kinds.items() if kind == "untrained_numeric_angle"]
    write_json(destination / "status.json", {"status": "fitting_training_probes"})
    fit_states = extract_programs(model, datasets["train"], [(), *canonical.values()], args)
    projector = fit_projector(torch.cat([
        fit_states[()], *[fit_states[canonical[angle]] for angle in GROUP_ANGLES if angle not in extrapolated]
    ]), args.latent_dim, config["seed"])
    fit_source = project(fit_states[()], projector)
    if len(fit_source) < 2:
        raise ValueError("At least two training sources are required for identifiable probe fitting.")
    fit_targets = {angle: project(fit_states[program], projector) for angle, program in canonical.items()}
    operators = {angle: fit_operator(fit_source, target, args.ridge) for angle, target in fit_targets.items()}
    torch.save({"projector": projector, "operators": operators}, destination / "probes.pt")
    means = {angle: target.mean(0) for angle, target in fit_targets.items()}
    training_fit = [{"angle": angle, "kind": kinds[angle],
                     **probe_fit_metrics(operators[angle], fit_source, target, means[angle])}
                    for angle, target in fit_targets.items()]
    save_csv(destination / "training_probe_fit.csv", training_fit)
    fit_hashes = datasets["train"].digests
    del fit_states
    report = {
        "status": "complete", "group": "C24: multiples of 15 degrees modulo 360",
        "checkpoint_kind": "random_initialization_control" if initialization_control else "trained_selected_checkpoint",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "frozen_model": True, "fit_sources": len(fit_source), "fit_source_sha256": fit_hashes,
        "source_counts": {split: len(data.sources) for split, data in datasets.items()},
        "representation": "final contextual image tokens, spatially average-pooled to a 6x6 grid by default, then training-only PCA and whitening",
        "pool_grid": args.pool_grid, "raw_feature_dimensions": int(projector["mean"].numel()),
        "probe_dimensions": int(projector["scales"].numel()), "ridge": args.ridge,
        "operator_convention": "affine homogeneous column action; program [a,b] -> rho(b)rho(a)",
        "canonical_programs": {angle: list(program) for angle, program in canonical.items()},
        "angle_kinds": kinds, "extrapolated_angles": extrapolated,
        "reference_encoding": "same frozen encoder with both task slots masked; this branch is not a trained identity task",
        "interpretation": "Probes test a shared affine action on compressed hidden states. They are not network operators and do not establish a homomorphism of the full nonlinear model. Poor source-generalization or collapsed variance invalidates a favorable algebraic residual.",
        "fit_policy": "All projection and operator fitting uses training sources only. Primary effect operators use trained atomic or seen-composition tokenizations; held-out pairs are never fitted. Identity and inverse-angle probes are explicitly OOD numeric-feature queries.",
        "associativity_policy": "Regrouped independently fitted effect maps and equivalent two-token programs; not the tautological associativity of matrix multiplication. The model has no three-token interface.",
        "summary_aggregation": "Per-law macro means; controlled latent subsets can have different sample counts, saved in each CSV row.",
        "commutativity_policy": "Primary commutativity summaries exclude equal-angle pairs; their self-commutation is tautological. Full per-pair tables retain them as labeled controls.",
        "input_encoding_caveat": "Sin/cos angle features already contain group-related structure. The matched random-initialization analysis checks how much structure exists before training; independently fitted PCA bases are not aligned across checkpoints.",
        "training_probe_fit": training_fit, "splits": {},
    }
    summary = {"fit_sources": len(fit_source), "probe_dimensions": report["probe_dimensions"]}
    for split in ("val", "test"):
        write_json(destination / "status.json", {"status": "evaluating", "split": split})
        dataset = datasets[split]
        states = extract_programs(model, dataset, [(), *canonical.values(), *ALL_PAIRS], args)
        source = project(states[()], projector)
        tables = group_laws(operators, source, dataset.seen_pairs, extrapolated)
        tables["probe_fit"] = [{"angle": angle, "kind": kinds[angle],
                                **probe_fit_metrics(operators[angle], source, project(states[program], projector), means[angle])}
                               for angle, program in canonical.items()]
        tables["latent_program_consistency"], tables["latent_regrouping"] = latent_program_tests(dataset, states, projector, operators, canonical)
        summaries = {name: numeric_summary([row for row in rows if not row.get("tautological", False)])
                     for name, rows in tables.items()}
        for kind in ("atomic_trained", "composition_effect_trained", "untrained_numeric_angle"):
            summaries[f"probe_fit/{kind}"] = numeric_summary([
                row for row in tables["probe_fit"] if row["kind"] == kind
            ])
        for category in ("seen_pairs", "held_out_pairs"):
            for name in ("composition", "commutativity"):
                summaries[f"{category}/{name}"] = numeric_summary([
                    row for row in tables[name] if row["pair_split"] == category and not row.get("tautological", False)
                ])
            for subset in ("all", "raster_exact_and_identity_preserved"):
                rows = [row for row in tables["latent_program_consistency"] if row["pair_split"] == category and row["subset"] == subset]
                summaries[f"{category}/latent/{subset}"] = {**numeric_summary(rows), "source_program_pairs": sum(row["samples"] for row in rows)}
                summaries[f"{category}/latent_nontrivial_commutativity/{subset}"] = numeric_summary([
                    {key: value for key, value in row.items() if key.startswith("raw_reverse_")}
                    for row in rows if row["a"] != row["b"]
                ])
        for kind, flag in (("trained_effects", False), ("extrapolation", True)):
            summaries[f"full_c24_composition/{kind}"] = numeric_summary([row for row in tables["full_c24_composition"] if row["extrapolation"] == flag])
        for name, rows in tables.items():
            save_csv(destination / f"{split}_{name}.csv", rows)
        report["splits"][split] = {"samples": len(source), "summary": summaries, "tables": tables}
        summary[split] = summaries
    # These artifacts stay local; only numeric summary scores go to W&B.
    write_json(destination / "report.json", report)
    write_json(destination / "summary.json", summary)
    write_json(destination / "status.json", {"status": "complete", "report": str(destination / "report.json")})
    print(json.dumps({"status": "complete", "report": str(destination / "report.json"),
                      "source_counts": report["source_counts"], "probe_dimensions": report["probe_dimensions"]}), flush=True)
    return report, summary


def numeric_differences(trained, initial):
    result = {}
    for key in trained.keys() & initial.keys():
        left, right = trained[key], initial[key]
        if isinstance(left, dict) and isinstance(right, dict):
            result[key] = numeric_differences(left, right)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[key] = left - right
    return result


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-sources", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--pool-grid", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--skip-initialization-control", action="store_true")
    args = parser.parse_args()
    if min(args.fit_sources, args.latent_dim, args.pool_grid, args.batch_size) < 1 or args.ridge < 0 or args.poll_seconds < 1:
        parser.error("Sizes and polling interval must be positive; ridge must be nonnegative.")
    output = args.output.resolve()
    destination = output / "group_tests"
    destination.mkdir(exist_ok=True)
    with (destination / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(destination / "config.json", {**vars(args), "output": str(output)})
        (destination / "pid.txt").write_text(str(os.getpid()) + "\n")
        try:
            training = wait_for_training(output, destination, args.wait, args.poll_seconds)
            _, trained_summary = evaluate(output, destination, training, args)
            combined = {"trained": trained_summary}
            if not args.skip_initialization_control:
                write_json(destination / "status.json", {"status": "running_initialization_control"})
                baseline = destination / "initialization_control"
                baseline.mkdir(exist_ok=True)
                _, initial_summary = evaluate(output, baseline, training, args, initialization_control=True)
                combined["initialization_control"] = initial_summary
                combined["trained_minus_initialization"] = numeric_differences(trained_summary, initial_summary)
            write_json(destination / "comparison.json", combined)
            if args.log_wandb:
                write_json(destination / "status.json", {"status": "logging_metrics"})
                url = publish_metrics(output, combined)
                write_json(destination / "wandb.json", {"url": url, "payload": "numeric group-theory summaries only"})
            write_json(destination / "status.json", {"status": "complete", "report": str(destination / "report.json"),
                                                    "comparison": str(destination / "comparison.json")})
        except Exception as error:
            write_json(destination / "status.json", {"status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
