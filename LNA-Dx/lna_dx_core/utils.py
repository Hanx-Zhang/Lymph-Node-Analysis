from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def dx_root() -> Path:
    """Return the LNA-Dx package root that contains train.py and test.py."""

    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' key.")
    return state_dict


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_matching_weights(
    model: torch.nn.Module,
    weights_path: str | Path,
    *,
    strict: bool = False,
) -> tuple[int, int]:
    """Load checkpoint weights, tolerating DataParallel prefixes and head changes.

    Returns the number of loaded tensors and the number of tensors in the
    checkpoint after prefix normalization.
    """

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    incoming = strip_module_prefix(checkpoint_state_dict(checkpoint))
    current = unwrap_model(model).state_dict()

    if strict:
        unwrap_model(model).load_state_dict(incoming, strict=True)
        return len(incoming), len(incoming)

    matched = {
        key: value
        for key, value in incoming.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    current.update(matched)
    unwrap_model(model).load_state_dict(current)
    return len(matched), len(incoming)


class AverageMeter:
    """Track running averages for losses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
