from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset


Sample = Tuple[str, str, int]


def _binary_label(raw_label: str) -> int:
    return 0 if int(raw_label) < 1 else 1


def _read_fold_file(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Fold list not found: {path}")

    samples: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line {line_number} in {path}: expected '<image> <label>'.")
            samples.append((parts[0], _binary_label(parts[1])))
    return samples


def _fold_paths(folder: str | Path, folds: str) -> list[Path]:
    folder = Path(folder)
    return [folder / f"fold_{fold}.txt" for fold in folds.split("-") if fold]


def load_samples(image_list_dir: str | Path, folds: str, prior_list_dir: str | Path) -> list[Sample]:
    image_entries: list[tuple[str, int]] = []
    for fold_path in _fold_paths(image_list_dir, folds):
        print(f"loading image list from: {fold_path}")
        image_entries.extend(_read_fold_file(fold_path))

    prior_entries: list[tuple[str, int]] = []
    for fold_path in _fold_paths(prior_list_dir, folds):
        print(f"loading prior list from: {fold_path}")
        prior_entries.extend(_read_fold_file(fold_path))

    if len(image_entries) != len(prior_entries):
        raise ValueError(
            "Image and prior lists must contain the same number of rows: "
            f"{len(image_entries)} != {len(prior_entries)}."
        )

    samples: list[Sample] = []
    for index, ((image_path, image_label), (prior_path, prior_label)) in enumerate(
        zip(image_entries, prior_entries),
        start=1,
    ):
        if image_label != prior_label:
            raise ValueError(
                f"Label mismatch at row {index}: image label {image_label}, prior label {prior_label}."
            )
        samples.append((image_path, prior_path, image_label))
    return samples


def load_dx_volume(image_path: str | Path, prior_path: str | Path) -> np.ndarray:
    """Load a 3-channel diagnosis volume.

    Channel 0: lung-window CT normalized from -1000 to 600 HU.
    Channel 1: mediastinal-window CT normalized from -160 to 240 HU.
    Channel 2: anatomical prior label map divided by 26.
    """

    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    lung_window = np.clip((image + 1000.0) / 1600.0, 0.0, 1.0)
    mediastinal_window = np.clip((image + 160.0) / 400.0, 0.0, 1.0)

    prior = sitk.GetArrayFromImage(sitk.ReadImage(str(prior_path))).astype(np.float32)
    prior = prior / 26.0

    return np.stack((lung_window, mediastinal_window, prior), axis=-1)


def augment_volume(volume: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply the same lightweight 3D augmentations used by the original code."""

    for axis in (0, 1, 2):
        if rng.random() < 0.5:
            volume = np.flip(volume, axis=axis)

    for axes in ((0, 1), (0, 2), (1, 2)):
        if rng.random() < 0.5:
            volume = np.rot90(volume, k=rng.randint(0, 3), axes=axes)

    for axes in ((1, 0, 2, 3), (2, 1, 0, 3), (0, 2, 1, 3)):
        if rng.random() < 0.5:
            volume = np.transpose(volume, axes)

    return np.ascontiguousarray(volume)


class LymphNodeDiagnosisDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        augment: bool = False,
        seed: int = 0,
        return_paths: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.augment = augment
        self.seed = seed
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, prior_path, label = self.samples[index]
        volume = load_dx_volume(image_path, prior_path)

        if self.augment:
            volume = augment_volume(volume, random.Random(self.seed + index))

        volume = np.transpose(volume, (3, 0, 1, 2))
        tensor = torch.from_numpy(np.ascontiguousarray(volume)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_paths:
            return tensor, label_tensor, image_path, prior_path
        return tensor, label_tensor


def split_train_val(samples: Sequence[Sample], val_fraction: float, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")

    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    train_size = int(math.ceil((1.0 - val_fraction) * len(samples)))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    return train_samples, val_samples


def create_train_val_loaders(
    image_list_dir: str | Path,
    prior_list_dir: str | Path,
    folds: str,
    *,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    augment: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    samples = load_samples(image_list_dir, folds, prior_list_dir)
    train_samples, val_samples = split_train_val(samples, val_fraction, seed=seed)

    train_dataset = LymphNodeDiagnosisDataset(train_samples, augment=augment, seed=seed)
    val_dataset = LymphNodeDiagnosisDataset(val_samples, augment=False, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_inference_loader(
    image_list_dir: str | Path,
    prior_list_dir: str | Path,
    folds: str,
    *,
    num_workers: int = 0,
) -> DataLoader:
    samples = load_samples(image_list_dir, folds, prior_list_dir)
    dataset = LymphNodeDiagnosisDataset(samples, augment=False, return_paths=True)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
