"""Pool C1 atomic-pair results and compute layerwise OOD correlations."""

import argparse
import json
from pathlib import Path

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments import (
    PROTOCOL_VERSION, compute_correlations, write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiments", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        choices=range(1, 6))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/cogitao_c1_atomic_pairs_all"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="cogitao-c1-atomic-pair-algebra")
    args = parser.parse_args()
    rows, reports, signature = [], [], None
    for experiment in dict.fromkeys(args.experiments):
        path = args.input_root / f"cogitao_c1_experiment_{experiment}_atomic_pairs" / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["experiment"] != experiment or report["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"Incompatible report: {path}")
        settings = (report["fit"]["pool_grid"], tuple(report["fit"]["layers"]),
                    report["fit"]["ridge"], report["relation_tolerance"],
                    tuple(report["fit"]["resolution"]),
                    report["evaluation_settings"]["max_accuracy_images"])
        if signature is not None and signature != settings:
            raise ValueError(
                "Reports use different pooling, layers, ridge, tolerance, resolution, or accuracy caps")
        signature = settings
        rows.extend(report["evaluation"]["pair_layers"])
        reports.append(dict(experiment=experiment, report=str(path),
                            checkpoint_sha256=report["fit"]["checkpoint_sha256"]))
    correlations = compute_correlations(rows, "all")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_layer_metrics.csv", rows)
    write_csv(args.output_dir / "correlations.csv", correlations)
    (args.output_dir / "report.json").write_text(json.dumps(dict(
        protocol_version=PROTOCOL_VERSION, source_reports=reports,
        unit="experiment and ordered composition; splits and layers kept separate",
        correlations=correlations), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        group=args.wandb_group, name="c1-all-atomic-pair-correlations",
                        mode=args.wandb_mode,
                        config=dict(experiments=args.experiments, reports=reports))
        try:
            if correlations:
                fields = list(correlations[0])
                run.log({"pairs/pooled_ood_correlations": wandb.Table(columns=fields,
                    data=[[row[key] for key in fields] for row in correlations])})
            artifact = wandb.Artifact("c1-all-atomic-pair-correlations", type="analysis")
            for name in ("report.json", "correlations.csv", "pair_layer_metrics.csv"):
                artifact.add_file(str(args.output_dir / name), name=name)
            run.log_artifact(artifact)
        finally:
            run.finish()
    print(f"Wrote pooled C1 results to {args.output_dir}")


if __name__ == "__main__":
    main()
