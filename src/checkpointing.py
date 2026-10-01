from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

try:
    from lightning.pytorch import Callback
except ImportError:  # pragma: no cover
    from pytorch_lightning import Callback


class TrainingProgressCheckpoint(Callback):
    """Save rolling and fixed-progress checkpoints independently of metrics."""

    DEFAULT_MILESTONES: ClassVar[dict[str, float]] = {
        "025pct": 0.25,
        "050pct": 0.50,
        "075pct": 0.75,
        "100pct": 1.00,
    }

    def __init__(
        self,
        dirpath: str | Path,
        *,
        milestones: dict[str, float] | None = None,
        rolling_filename: str = "last.ckpt",
        start_filename: str = "milestone_start.ckpt",
        training_end_filename: str = "training_end.ckpt",
        exception_filename: str = "interrupted.ckpt",
        save_on_exception: bool = True,
        save_weights_only: bool = False,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.milestones = dict(milestones or self.DEFAULT_MILESTONES)
        if not self.milestones:
            raise ValueError(
                "milestones must contain at least one checkpoint fraction."
            )
        for name, fraction in self.milestones.items():
            if not name or not 0.0 < float(fraction) <= 1.0:
                raise ValueError(
                    "Milestone names must be non-empty and fractions must be in (0, 1]."
                )
        self.rolling_filename = str(rolling_filename)
        self.start_filename = str(start_filename)
        self.training_end_filename = str(training_end_filename)
        self.exception_filename = str(exception_filename)
        self.save_on_exception = bool(save_on_exception)
        self.save_weights_only = bool(save_weights_only)
        self.saved_milestones: set[str] = set()

    @property
    def state_key(self) -> str:
        return f"{self.__class__.__qualname__}[{self.dirpath}]"

    def state_dict(self) -> dict[str, Any]:
        return {"saved_milestones": sorted(self.saved_milestones)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.saved_milestones = set(state_dict.get("saved_milestones", []))

    def _save(self, trainer: Any, filename: str) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            str(self.dirpath / filename),
            weights_only=self.save_weights_only,
        )

    def _milestone_epochs(self, max_epochs: int) -> dict[str, int]:
        if max_epochs <= 0:
            raise ValueError(
                "TrainingProgressCheckpoint requires trainer.max_epochs > 0."
            )
        return {
            name: max(1, math.ceil(max_epochs * float(fraction)))
            for name, fraction in self.milestones.items()
        }

    def on_train_start(self, trainer: Any, _pl_module: Any) -> None:
        # A legacy checkpoint has no callback state, so resuming it creates a
        # start snapshot in the new run. Resuming this same managed run restores
        # the "start" marker and does not replace its existing snapshot.
        if "start" not in self.saved_milestones:
            self.saved_milestones.add("start")
            self._save(trainer, self.start_filename)

    def on_train_epoch_end(self, trainer: Any, _pl_module: Any) -> None:
        completed_epochs = int(trainer.current_epoch) + 1
        for name, target_epoch in self._milestone_epochs(
            int(trainer.max_epochs)
        ).items():
            if completed_epochs == target_epoch and name not in self.saved_milestones:
                self.saved_milestones.add(name)
                self._save(trainer, f"milestone_{name}.ckpt")

        # This is deliberately unconditional. In particular, it does not
        # depend on whether a validation metric entered the retained top-k.
        self._save(trainer, self.rolling_filename)

    def on_train_end(self, trainer: Any, _pl_module: Any) -> None:
        self._save(trainer, self.training_end_filename)
        self._save(trainer, self.rolling_filename)

    def on_exception(
        self, trainer: Any, _pl_module: Any, _exception: BaseException
    ) -> None:
        if not self.save_on_exception:
            return
        try:
            self._save(trainer, self.exception_filename)
        except Exception:  # noqa: BLE001
            # Never hide the original training failure with a secondary
            # checkpoint-writing failure.
            return
