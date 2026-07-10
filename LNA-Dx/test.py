from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from lna_dx_core.data import create_inference_loader
from lna_dx_core.engine import classifier_weight, gs_cam, register_feature_hook
from lna_dx_core.model_factory import generate_model
from lna_dx_core.utils import dx_root, ensure_dir, load_matching_weights


REPO_ROOT = dx_root().parent
DEFAULT_DX_VOI_ROOT = REPO_ROOT / "LNA-Outputs" / "03_voi_for_lna_dx"
DEFAULT_IMAGE_VOI_DIR = DEFAULT_DX_VOI_ROOT / "img_VOI"
DEFAULT_PRIOR_VOI_DIR = DEFAULT_DX_VOI_ROOT / "seg_VOI"
DEFAULT_DX_OUTPUT_ROOT = REPO_ROOT / "LNA-Outputs" / "04_lna_dx"
DEFAULT_OUTPUT_FILE = DEFAULT_DX_OUTPUT_ROOT / "Predictions_Dx.xlsx"
DEFAULT_GS_CAM1_DIR = DEFAULT_DX_OUTPUT_ROOT / "gs-cam1_VOI"
DEFAULT_GT_TABLE = REPO_ROOT / "LNA-Inputs" / "case_info" / "case_table_local.csv"
GT_COLUMN_CANDIDATES = ("n_stage", "n-stage", "nstage", "n stage", "gt", "stage")


def case_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def normalize_n_stage(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return ""
    if text.startswith("N"):
        return text
    try:
        numeric = int(float(text))
    except ValueError:
        return text
    return f"N{numeric}"


def resolve_table_column(frame: pd.DataFrame, requested: str | None, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if requested:
        key = requested.strip().lower()
        if key not in columns:
            raise ValueError(f"Column '{requested}' not found in GT table. Available columns: {list(frame.columns)}")
        return columns[key]
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def read_gt_table(gt_table: Path | None, case_id_column: str, gt_column: str | None) -> dict[str, str]:
    if gt_table is None:
        return {}
    if not gt_table.exists():
        raise FileNotFoundError(f"GT table not found: {gt_table}")
    if gt_table.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(gt_table)
    else:
        frame = pd.read_csv(gt_table)
    if frame.empty:
        return {}

    case_col = resolve_table_column(frame, case_id_column, ("case_id", "case", "id"))
    if case_col is None:
        raise ValueError("GT table must contain a case_id column.")
    stage_col = resolve_table_column(frame, gt_column, GT_COLUMN_CANDIDATES)
    if stage_col is None:
        raise ValueError(
            "GT table must contain one N-stage column. "
            f"Accepted names: {', '.join(GT_COLUMN_CANDIDATES)}."
        )

    gt_by_case: dict[str, str] = {}
    for _, row in frame.iterrows():
        case_id = str(row[case_col]).strip()
        if case_id:
            gt_by_case[case_id] = normalize_n_stage(row[stage_col])
    return gt_by_case


def save_gs_cam1_nii(cam1_map: np.ndarray, image_path: str | Path, output_dir: Path) -> Path:
    image_path = Path(image_path)
    output_path = output_dir / image_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    template_image = sitk.ReadImage(str(image_path))
    template_array_shape = sitk.GetArrayFromImage(template_image).shape
    cam_array = np.transpose(cam1_map, (2, 0, 1))
    if cam_array.shape != template_array_shape:
        raise ValueError(
            f"GS-CAM1 shape {cam_array.shape} does not match VOI image shape {template_array_shape} for {image_path}."
        )

    cam_image = sitk.GetImageFromArray(np.rint(cam_array).astype(np.uint8), isVector=False)
    cam_image.CopyInformation(template_image)
    sitk.WriteImage(cam_image, str(output_path), True)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LNA-Dx inference.")

    parser.add_argument("--savedir", default="R18", help="Checkpoint/result subdirectory name.")
    parser.add_argument("--model", default="resnet_groupnorm", choices=["resnet_groupnorm"])
    parser.add_argument("--model-depth", type=int, default=18, choices=[10, 18, 34, 50, 101, 152, 200])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--resnet-shortcut", default="B", choices=["A", "B"])

    parser.add_argument("--image-voi-dir", type=Path, default=DEFAULT_IMAGE_VOI_DIR)
    parser.add_argument("--prior-voi-dir", type=Path, default=DEFAULT_PRIOR_VOI_DIR)
    parser.add_argument("--label-table", type=Path, default=None, help="Optional CSV/XLSX with case_id and label.")
    parser.add_argument("--label-column", default=None, help="Optional diagnosis label column name in --label-table.")
    parser.add_argument("--case-id-column", default="case_id", help="Case id column name in --label-table.")
    parser.add_argument(
        "--gt-table",
        type=Path,
        default=DEFAULT_GT_TABLE if DEFAULT_GT_TABLE.exists() else None,
        help=f"Optional CSV/XLSX with case_id and N-stage for the output gt column. Default: {DEFAULT_GT_TABLE}",
    )
    parser.add_argument("--gt-column", default=None, help="Optional N-stage column name in --gt-table.")
    parser.add_argument("--default-label", type=int, default=0, help="Placeholder label used when no label table is provided.")
    parser.add_argument("--image-list-dir", type=Path, default=None, help="Legacy fold-list image directory.")
    parser.add_argument("--prior-list-dir", type=Path, default=None, help="Legacy fold-list prior directory.")
    parser.add_argument("--test-folds", default="1", help="Legacy fold ids joined by '-', for example '1-2'.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use '' for the runtime default.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to trained LNA-Dx checkpoint. Default: checkpoints/<savedir>/weights_dx.pth.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=f"Prediction table path. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--save-cam-dir",
        type=Path,
        default=None,
        help="Optional directory for CAM0/CAM1 .npy files.",
    )
    parser.add_argument(
        "--gs-cam1-dir",
        type=Path,
        default=DEFAULT_GS_CAM1_DIR,
        help=f"Directory for GS-CAM1 VOI .nii.gz outputs. Default: {DEFAULT_GS_CAM1_DIR}",
    )
    return parser.parse_args()


def save_table(rows: list[dict], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    df = pd.DataFrame(rows)
    if output_file.suffix.lower() == ".csv":
        df.to_csv(output_file, index=False)
    else:
        df.to_excel(output_file, index=False)


def run_inference(args: argparse.Namespace) -> None:
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.weights is None:
        args.weights = dx_root() / "checkpoints" / args.savedir / "weights_dx.pth"
    if args.output_file is None:
        args.output_file = DEFAULT_OUTPUT_FILE

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    gt_by_case = read_gt_table(args.gt_table, args.case_id_column, args.gt_column)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = generate_model(args)
    loaded, total = load_matching_weights(model, args.weights, strict=True)
    print(f"Loaded {loaded}/{total} tensors from {args.weights}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    loader = create_inference_loader(
        image_list_dir=args.image_list_dir,
        prior_list_dir=args.prior_list_dir,
        folds=args.test_folds,
        image_voi_dir=args.image_voi_dir,
        prior_voi_dir=args.prior_voi_dir,
        label_table=args.label_table,
        label_column=args.label_column,
        case_id_column=args.case_id_column,
        default_label=args.default_label,
        num_workers=args.num_workers,
    )

    if args.save_cam_dir is not None:
        ensure_dir(args.save_cam_dir)
    if args.gs_cam1_dir is not None:
        ensure_dir(args.gs_cam1_dir)

    rows: list[dict] = []
    feature_blobs: list[torch.Tensor] = []
    hook = register_feature_hook(model, feature_blobs)

    try:
        with torch.no_grad():
            for batch_index, (images, labels, image_paths, prior_paths) in enumerate(loader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                feature_blobs.clear()

                logits = model(images)
                if not feature_blobs:
                    raise RuntimeError("Feature hook did not capture relu_out activations.")
                probabilities = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

                weights = classifier_weight(model)
                cam0 = gs_cam(feature_blobs[0], weights, 0).unsqueeze(1)
                cam1 = gs_cam(feature_blobs[0], weights, 1).unsqueeze(1)
                cam0 = F.interpolate(cam0, size=images.shape[-3:], mode="trilinear", align_corners=True)
                cam1 = F.interpolate(cam1, size=images.shape[-3:], mode="trilinear", align_corners=True)
                cam0_map = np.clip(cam0.squeeze().detach().cpu().numpy(), 0.0, 1.0)
                cam1_map = np.clip(cam1.squeeze().detach().cpu().numpy(), 0.0, 1.0)

                if args.save_cam_dir is not None:
                    case_id = case_id_from_path(image_paths[0])
                    np.save(args.save_cam_dir / f"{batch_index:04d}_{case_id}_CAM0.npy", cam0_map)
                    np.save(args.save_cam_dir / f"{batch_index:04d}_{case_id}_CAM1.npy", cam1_map)

                case_id = case_id_from_path(image_paths[0])
                gt = gt_by_case.get(case_id, "")
                gs_cam1_path = None
                if args.gs_cam1_dir is not None:
                    gs_cam1_path = save_gs_cam1_nii(cam1_map, image_paths[0], args.gs_cam1_dir)

                rows.append(
                    {
                        "case_index": batch_index,
                        "case_id": case_id,
                        "gt": gt,
                        "prediction": str(probabilities.tolist()),
                        "image_path": image_paths[0],
                        "prior_path": prior_paths[0],
                        "GS-CAM1_path": str(gs_cam1_path) if gs_cam1_path is not None else "",
                    }
                )
    finally:
        hook.remove()

    save_table(rows, args.output_file)
    print(f"Saved predictions: {args.output_file}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
