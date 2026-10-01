from pathlib import Path

import yaml


SWEEP_ROOT = Path("benchmarks/cogitao_slot_attention")


def test_setting_sweeps_cover_all_five_experiments_once():
    for setting in (1, 2):
        path = SWEEP_ROOT / f"sweep_setting{setting}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["method"] == "grid"
        assert config["program"] == "benchmarks.cogitao_slot_attention.run"
        assert config["parameters"]["setting"]["value"] == setting
        assert config["parameters"]["experiment"]["values"] == [1, 2, 3, 4, 5]
        assert config["command"] == [
            "${env}",
            "${interpreter}",
            "-m",
            "${program}",
            "${args}",
        ]
