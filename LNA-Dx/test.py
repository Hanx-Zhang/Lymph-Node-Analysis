from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lna_dx.data import create_inference_loader
from lna_dx.model_factory import generate_model
from lna_dx.training import classifier_weight, gs_cam, register_feature_hook
from lna_dx.utils import default_external_data_root, dx_root, ensure_dir, load_matching_weights


def default_petct_image_lists() -> Path:
    return default_external_data_root() / "PETCTDx" / "sizeX" / "3D_img_resam_voi_2-fold_notRandom_txt"


def default_petct_prior_lists() -> Path:
    return default_external_data_root() / "PETCTDx" / "sizeX" / "3D_seg_resam_voi_2-fold_notRandom_txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LNA-Dx inference.")

    parser.add_argument("--savedir", default="R18", help="Checkpoint/result subdirectory name.")
    parser.add_argument("--model", default="resnet_groupnorm", choices=["resnet_groupnorm"])
    parser.add_argument("--model-depth", type=int, default=18, choices=[10, 18, 34, 50, 101, 152, 200])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--resnet-shortcut", default="B", choices=["A", "B"])

    parser.add_argument("--image-list-dir", type=Path, default=default_petct_image_lists())
    parser.add_argument("--prior-list-dir", type=Path, default=default_petct_prior_lists())
    parser.add_argument("--test-folds", default="1", help="Fold ids joined by '-', for example '1-2'.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use '' for the runtime default.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to trained LNA-Dx checkpoint. Default: checkpoints/<savedir>/weights_cls.pth.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Prediction table path. Supported extensions: .xlsx, .csv.",
    )
    parser.add_argument(
        "--save-cam-dir",
        type=Path,
        default=None,
        help="Optional directory for CAM0/CAM1 .npy files.",
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
        args.weights = dx_root() / "checkpoints" / args.savedir / "weights_cls.pth"
    if args.output_file is None:
        args.output_file = dx_root() / "results" / args.savedir / "PETCTDx_Predictions_Cls.xlsx"

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

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
        args.image_list_dir,
        args.prior_list_dir,
        args.test_folds,
        num_workers=args.num_workers,
    )

    if args.save_cam_dir is not None:
        ensure_dir(args.save_cam_dir)

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
                cam0_np = np.clip((cam0.squeeze().detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                cam1_np = np.clip((cam1.squeeze().detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)

                if args.save_cam_dir is not None:
                    stem = Path(image_paths[0]).stem
                    np.save(args.save_cam_dir / f"{batch_index:04d}_{stem}_CAM0.npy", cam0_np)
                    np.save(args.save_cam_dir / f"{batch_index:04d}_{stem}_CAM1.npy", cam1_np)

                rows.append(
                    {
                        "case_index": batch_index,
                        "image_path": image_paths[0],
                        "prior_path": prior_paths[0],
                        "label": int(labels.item()),
                        "prob_non_metastatic": float(probabilities[0]),
                        "prob_metastatic": float(probabilities[1]),
                        "prediction": int(np.argmax(probabilities)),
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
