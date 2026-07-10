from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from volumentations import Compose, Flip, RandomRotate90, Transpose


Sample = Tuple[str, str, int]


def get_augmentation():
    return Compose(
        [
            Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            RandomRotate90((0, 1), p=0.5),
            RandomRotate90((0, 2), p=0.5),
            RandomRotate90((1, 2), p=0.5),
            Transpose((1, 0, 2, 3), p=0.5),
            Transpose((2, 1, 0, 3), p=0.5),
            Transpose((0, 2, 1, 3), p=0.5),
        ],
        p=0.8,
    )


def _binary_label(raw_label: str | int | float) -> int:
    text = str(raw_label).strip().lower()
    if text in ("n0", "stage0", "stage 0"):
        return 0
    if text in ("n1", "n2", "n3", "stage1", "stage2", "stage3", "stage 1", "stage 2", "stage 3"):
        return 1
    if text in ("positive", "pos", "metastatic", "metastasis", "yes", "true"):
        return 1
    if text in ("negative", "neg", "non-metastatic", "nonmetastatic", "no", "false"):
        return 0
    return 0 if int(float(text)) < 1 else 1


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


def case_id_from_nii_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


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


LABEL_COLUMN_CANDIDATES = (
    "label",
    "dx_label",
    "diagnosis",
    "metastasis",
    "metastatic",
    "n_stage",
    "n-stage",
    "nstage",
    "stage",
    "gt",
    "target",
    "y",
)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label table not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _resolve_column(
    frame: pd.DataFrame,
    requested: Optional[str],
    candidates: Sequence[str],
) -> Optional[str]:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if requested:
        key = requested.strip().lower()
        if key not in columns:
            raise ValueError(f"Column '{requested}' not found in label table. Available columns: {list(frame.columns)}")
        return columns[key]
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def read_label_table(
    label_table: str | Path,
    *,
    label_column: Optional[str] = None,
    case_id_column: str = "case_id",
    require_labels: bool = True,
) -> dict[str, int]:
    frame = _read_table(label_table)
    if frame.empty:
        raise ValueError(f"Label table is empty: {label_table}")

    case_col = _resolve_column(frame, case_id_column, ("case_id", "case", "id"))
    if case_col is None:
        raise ValueError("Label table must contain a case_id column.")

    label_col = _resolve_column(frame, label_column, LABEL_COLUMN_CANDIDATES)
    if label_col is None:
        if require_labels:
            raise ValueError(
                "Label table must contain one diagnosis label column. "
                f"Accepted names: {', '.join(LABEL_COLUMN_CANDIDATES)}."
            )
        return {}

    labels: dict[str, int] = {}
    for _, row in frame.iterrows():
        case_id = str(row[case_col]).strip()
        if not case_id:
            continue
        labels[case_id] = _binary_label(row[label_col])
    return labels


def load_samples_from_voi_dirs(
    image_voi_dir: str | Path,
    prior_voi_dir: str | Path,
    *,
    label_table: str | Path | None = None,
    label_column: Optional[str] = None,
    case_id_column: str = "case_id",
    default_label: int | None = 0,
    require_labels: bool = False,
) -> list[Sample]:
    image_voi_dir = Path(image_voi_dir)
    prior_voi_dir = Path(prior_voi_dir)
    if not image_voi_dir.is_dir():
        raise FileNotFoundError(f"Image VOI folder not found: {image_voi_dir}")
    if not prior_voi_dir.is_dir():
        raise FileNotFoundError(f"Prior VOI folder not found: {prior_voi_dir}")

    labels: dict[str, int] = {}
    if label_table is not None:
        labels = read_label_table(
            label_table,
            label_column=label_column,
            case_id_column=case_id_column,
            require_labels=require_labels,
        )
    elif require_labels:
        raise ValueError("Training from VOI folders requires --label-table with diagnosis labels.")

    image_files = sorted([path for path in image_voi_dir.iterdir() if path.name.endswith(".nii.gz")])
    if not image_files:
        raise FileNotFoundError(f"No .nii.gz files found in {image_voi_dir}")

    samples: list[Sample] = []
    for image_path in image_files:
        case_id = case_id_from_nii_name(image_path.name)
        prior_path = prior_voi_dir / image_path.name
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing prior VOI for {case_id}: {prior_path}")

        if case_id in labels:
            label = labels[case_id]
        elif require_labels:
            raise ValueError(f"Missing diagnosis label for case_id '{case_id}'.")
        elif default_label is not None:
            label = int(default_label)
        else:
            raise ValueError(f"Missing diagnosis label for case_id '{case_id}'.")
        samples.append((str(image_path), str(prior_path), label))

    print(f"loaded {len(samples)} VOI samples from {image_voi_dir} and {prior_voi_dir}")
    return samples


def load_dx_volume(image_path: str | Path, prior_path: str | Path) -> np.ndarray:
    """Load a 3-channel diagnosis volume.

    Channel 0: lung-window CT normalized from -1000 to 600 HU.
    Channel 1: mediastinal-window CT normalized from -160 to 240 HU.
    Channel 2: anatomical prior label map divided by 26.
    """

    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    lung_window = np.clip((image + 1000.0) / 1600.0, 0.0, 1.0)
    mediastinal_window = np.clip((image + 160.0) / 400.0, 0.0, 1.0)

    prior = sitk.GetArrayFromImage(sitk.ReadImage(str(prior_path)))
    prior = prior / 26.0

    return np.stack((lung_window, mediastinal_window, prior), axis=-1)


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
        self.data_transforms = get_augmentation()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, prior_path, label = self.samples[index]
        volume = load_dx_volume(image_path, prior_path)

        if self.augment:
            data = {
                "image": volume,
                "image_XX": volume.copy(),
            }
            volume = self.data_transforms(**data)["image"]

        volume = np.transpose(volume, (3, 1, 2, 0))
        tensor = torch.from_numpy(np.ascontiguousarray(volume)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_paths:
            return tensor, label_tensor, image_path, prior_path
        return tensor, label_tensor


def split_train_val(samples: Sequence[Sample], val_fraction: float, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")

    samples = list(samples)
    train_size = int(math.ceil((1.0 - val_fraction) * len(samples)))
    val_size = len(samples) - train_size
    torch.manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(samples, [train_size, val_size])
    return list(train_subset), list(val_subset)


def create_balanced_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler | None:
    labels = [sample[2] for sample in samples]
    class_counts = Counter(labels)
    if len(class_counts) < 2:
        print(f"Balanced sampling skipped because only one class is present: {dict(class_counts)}")
        return None

    sample_weights = [1.0 / class_counts[label] for label in labels]
    print(f"Using balanced sampling with class counts: {dict(sorted(class_counts.items()))}")
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_train_val_loaders(
    image_list_dir: str | Path | None,
    prior_list_dir: str | Path | None,
    folds: str,
    *,
    image_voi_dir: str | Path | None = None,
    prior_voi_dir: str | Path | None = None,
    label_table: str | Path | None = None,
    label_column: Optional[str] = None,
    case_id_column: str = "case_id",
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    augment: bool,
    balanced_sampling: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if image_list_dir is not None and prior_list_dir is not None:
        samples = load_samples(image_list_dir, folds, prior_list_dir)
    else:
        if image_voi_dir is None or prior_voi_dir is None:
            raise ValueError("Provide either --image-list-dir/--prior-list-dir or --image-voi-dir/--prior-voi-dir.")
        samples = load_samples_from_voi_dirs(
            image_voi_dir,
            prior_voi_dir,
            label_table=label_table,
            label_column=label_column,
            case_id_column=case_id_column,
            default_label=None,
            require_labels=True,
        )
    train_samples, val_samples = split_train_val(samples, val_fraction, seed=seed)

    train_dataset = LymphNodeDiagnosisDataset(train_samples, augment=augment, seed=seed)
    val_dataset = LymphNodeDiagnosisDataset(val_samples, augment=False, seed=seed)
    train_sampler = create_balanced_sampler(train_samples) if balanced_sampling else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    image_list_dir: str | Path | None = None,
    prior_list_dir: str | Path | None = None,
    folds: str = "1",
    *,
    image_voi_dir: str | Path | None = None,
    prior_voi_dir: str | Path | None = None,
    label_table: str | Path | None = None,
    label_column: Optional[str] = None,
    case_id_column: str = "case_id",
    default_label: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    if image_list_dir is not None and prior_list_dir is not None:
        samples = load_samples(image_list_dir, folds, prior_list_dir)
    else:
        if image_voi_dir is None or prior_voi_dir is None:
            raise ValueError("Provide either list folders or VOI folders for inference.")
        samples = load_samples_from_voi_dirs(
            image_voi_dir,
            prior_voi_dir,
            label_table=label_table,
            label_column=label_column,
            case_id_column=case_id_column,
            default_label=default_label,
            require_labels=False,
        )
    dataset = LymphNodeDiagnosisDataset(samples, augment=False, return_paths=True)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
