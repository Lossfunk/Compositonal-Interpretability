from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.cogitao.cogitao import (
    ARC_PALETTE,
    TASK_TO_ID,
    COGITAODataset,
)


def _write_split(root: Path, split: str, rows: list[dict]) -> None:
    path = root / "exp_setting_1" / "experiment_1" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_cogitao_renders_grids_targets_and_collatable_metadata(tmp_path):
    rows = [
        {
            "input": [[0, 1], [2, 3]],
            "output": [[3, 2], [1, 0]],
            "transformation_suite": ["rot90", "translate_up"],
            "task_key": "task-a",
        },
        {
            "input": [[4, 5], [6, 7]],
            "output": [[7, 6], [5, 4]],
            "transformation_suite": ["mirror_horizontal"],
            "task_key": "task-b",
        },
    ]
    _write_split(tmp_path, "train", rows)
    dataset = COGITAODataset(
        {
            "root": tmp_path,
            "setting": 1,
            "experiment": 1,
            "split": "train",
            "image_size": [4, 4],
        }
    )

    sample = dataset[0]
    assert sample["images"].shape == (3, 4, 4)
    assert sample["target_images"].shape == (3, 4, 4)
    torch.testing.assert_close(sample["images"][:, 0, 0], ARC_PALETTE[0])
    torch.testing.assert_close(sample["images"][:, 0, 3], ARC_PALETTE[1])
    assert sample["metadata"]["transformation_suite"] == "rot90 -> translate_up"
    assert sample["metadata"]["composition_depth"] == 2
    assert sample["task_tokens"].tolist() == [
        TASK_TO_ID["rot90"],
        TASK_TO_ID["translate_up"],
        0,
    ]
    assert sample["task_token_mask"].tolist() == [True, True, False]
    assert sample["target_grid"].shape == (4, 4)

    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 3, 4, 4)
    assert batch["metadata"]["composition_depth"].tolist() == [2, 1]


def test_cogitao_can_skip_unused_targets_and_metadata(tmp_path):
    _write_split(
        tmp_path,
        "val_ood",
        [
            {
                "input": [[0]],
                "output": [[1]],
                "transformation_suite": ["change_color"],
                "task_key": "task-c",
            }
        ],
    )
    dataset = COGITAODataset(
        {
            "root": tmp_path,
            "split": "val_ood",
            "image_size": 1,
            "return_target": False,
            "return_metadata": False,
            "return_task_tokens": False,
        }
    )
    assert set(dataset[0]) == {"images"}
