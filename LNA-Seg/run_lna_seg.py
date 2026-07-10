import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INITIAL_CWD = Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_INPUT_DIR = REPO_ROOT / "LNA-Inputs" / "case_data_nii"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "LNA-Outputs" / "02_lna_seg"
DEFAULT_WORK_DIR = REPO_ROOT / "LNA-Outputs" / "01_preprocess"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "nnUnet_trained_models"
DEFAULT_TASK_NAME = "Task224_lymph_Labels25_Tr82_Ts20_wTotalOnehot5"

from lna_seg_utils.utils import (
    ImageResample_to_newSize,
    load_itk_image_with_sampling,
    multi_label_connected_components_delate_small,
)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = INITIAL_CWD / path
    return path.resolve()


def case_id_from_nii_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def load_itk_image_new(filename: str):
    itkimage = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itkimage)
    numpy_origin = list(reversed(itkimage.GetOrigin()))
    numpy_spacing = list(reversed(itkimage.GetSpacing()))
    numpy_direction = list(reversed(itkimage.GetDirection()))
    return numpy_image, numpy_origin, numpy_spacing, numpy_direction


def restore_image_via_box_from_json(origin_shape, image, lungbox):
    restored = np.zeros(shape=origin_shape, dtype=np.uint8)
    crop_coords = []
    for dim in range(3):
        start = max(0, int(lungbox[dim * 2]))
        end = min(origin_shape[dim], int(lungbox[dim * 2 + 1]))
        if start >= end:
            raise ValueError(f"Invalid crop box on dim {dim}: [{start}, {end}]")
        crop_coords.append((start, end))

    target_shape = tuple(end - start for start, end in crop_coords)
    if tuple(image.shape) != target_shape:
        raise ValueError(f"Prediction shape {tuple(image.shape)} does not match crop box shape {target_shape}")

    restored[
        crop_coords[0][0] : crop_coords[0][1],
        crop_coords[1][0] : crop_coords[1][1],
        crop_coords[2][0] : crop_coords[2][1],
    ] = image
    return restored


def parse_folds(value):
    if value is None or value == "None":
        return None
    if value == "all":
        return "all"
    if isinstance(value, list):
        if len(value) == 1 and value[0] in ("None", "all"):
            return None if value[0] == "None" else "all"
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split(","))


def to_sitk_tuple(values):
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(reversed(values))
    return tuple(reversed(values.tolist()))


def save_w_post_del_total(image, image_prior, filename, origin, spacing, old_spacing, direction, old_size, islabel=True):
    origin = to_sitk_tuple(origin)
    spacing = to_sitk_tuple(spacing)
    old_spacing = to_sitk_tuple(old_spacing)
    direction = to_sitk_tuple(direction)
    old_size = to_sitk_tuple(old_size)

    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    new_image_sitk, _ = ImageResample_to_newSize(
        itkimage, newSize=old_size, newSpacing=old_spacing, is_label=islabel
    )

    seg_array_del_small, _, _ = multi_label_connected_components_delate_small(
        sitk.GetArrayFromImage(new_image_sitk), new_image_sitk.GetSpacing()
    )
    new_image_sitk_post = sitk.GetImageFromArray(seg_array_del_small, isVector=False)
    new_image_sitk_post.SetSpacing(old_spacing)
    new_image_sitk_post.SetOrigin(origin)
    new_image_sitk_post.SetDirection(direction)

    itkimage_prior = sitk.GetImageFromArray(image_prior, isVector=False)
    itkimage_prior.SetSpacing(spacing)
    itkimage_prior.SetOrigin(origin)
    itkimage_prior.SetDirection(direction)
    new_image_prior_sitk, _ = ImageResample_to_newSize(
        itkimage_prior, newSize=old_size, newSpacing=old_spacing, is_label=islabel
    )

    prior_array = sitk.GetArrayFromImage(new_image_prior_sitk)
    post_array = sitk.GetArrayFromImage(new_image_sitk_post)
    post_array[prior_array == 1] = 0
    post_array_del_small, _, _ = multi_label_connected_components_delate_small(post_array, old_spacing)

    itkimage_out = sitk.GetImageFromArray(post_array_del_small, isVector=False)
    itkimage_out.SetSpacing(old_spacing)
    itkimage_out.SetOrigin(origin)
    itkimage_out.SetDirection(direction)
    sitk.WriteImage(itkimage_out, str(filename), True)


def resolve_task_name(task_name: str, model_name: str, network_training_output_dir: str) -> str:
    if task_name.startswith("Task"):
        return task_name

    task_id = int(task_name)
    model_task_root = Path(network_training_output_dir) / model_name
    candidates = sorted(model_task_root.glob(f"Task{task_id:03d}_*"))
    if candidates:
        return candidates[0].name

    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

    return convert_id_to_task_name(task_id)


def run_nnunet_prediction(args, nnunet_input_dir: Path, nnunet_output_dir: Path, model_dir: Path):
    os.environ["RESULTS_FOLDER"] = str(model_dir)
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from batchgenerators.utilities.file_and_folder_operations import isdir, join
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import default_cascade_trainer, default_plans_identifier, default_trainer, network_training_output_dir

    task_name = resolve_task_name(args.task_name, args.model, network_training_output_dir)

    trainer = args.trainer_class_name or default_trainer
    cascade_trainer = args.cascade_trainer_class_name or default_cascade_trainer
    plans_identifier = args.plans_identifier or default_plans_identifier
    folds = parse_folds(args.folds)
    lowres_segmentations = None if args.lowres_segmentations in (None, "None") else args.lowres_segmentations
    all_in_gpu = None if args.all_in_gpu == "None" else args.all_in_gpu.lower() == "true"

    if args.model == "3d_cascade_fullres" and lowres_segmentations is None:
        lowres_model_folder = join(network_training_output_dir, "3d_lowres", task_name, trainer + "__" + plans_identifier)
        assert isdir(lowres_model_folder), f"Lowres model output folder not found: {lowres_model_folder}"
        lowres_output_folder = join(str(nnunet_output_dir), "3d_lowres_predictions")
        predict_from_folder(
            lowres_model_folder,
            str(nnunet_input_dir),
            lowres_output_folder,
            folds,
            False,
            args.num_threads_preprocessing,
            args.num_threads_nifti_save,
            None,
            args.part_id,
            args.num_parts,
            not args.disable_tta,
            mixed_precision=not args.disable_mixed_precision,
            overwrite_existing=args.overwrite_existing,
            checkpoint_name=args.checkpoint_name,
        )
        lowres_segmentations = lowres_output_folder
        trainer = cascade_trainer

    model_folder = join(network_training_output_dir, args.model, task_name, trainer + "__" + plans_identifier)
    assert isdir(model_folder), f"Model output folder not found: {model_folder}"

    predict_from_folder(
        model_folder,
        str(nnunet_input_dir),
        str(nnunet_output_dir),
        folds,
        args.save_npz,
        args.num_threads_preprocessing,
        args.num_threads_nifti_save,
        lowres_segmentations,
        args.part_id,
        args.num_parts,
        not args.disable_tta,
        mixed_precision=not args.disable_mixed_precision,
        overwrite_existing=args.overwrite_existing,
        mode=args.mode,
        overwrite_all_in_gpu=all_in_gpu,
        step_size=args.step_size,
        checkpoint_name=args.checkpoint_name,
    )


def postprocess_predictions(raw_input_dir: Path, preprocess_dir: Path, nnunet_output_dir: Path, output_dir: Path):
    bbox_path = preprocess_dir / "lung_bbox_dict.json"
    prior_full_dir = preprocess_dir / "prior_full"
    if not bbox_path.is_file():
        raise FileNotFoundError(f"Missing {bbox_path}")
    if not prior_full_dir.is_dir():
        raise FileNotFoundError(f"Missing {prior_full_dir}")

    with bbox_path.open("r", encoding="utf-8") as f:
        lung_bbox_dict = json.load(f)

    w_post_del_total_dir = output_dir / "wPost_delTotal"
    w_post_del_total_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted([p for p in nnunet_output_dir.iterdir() if p.name.endswith(".nii.gz")])
    if not pred_files:
        raise FileNotFoundError(f"No nnUNet predictions found in {nnunet_output_dir}")

    for pred_file in pred_files:
        case_id = case_id_from_nii_name(pred_file.name)
        raw_image_path = raw_input_dir / pred_file.name
        prior_path = prior_full_dir / pred_file.name
        if not raw_image_path.is_file():
            raise FileNotFoundError(f"Missing raw image for {case_id}: {raw_image_path}")
        if not prior_path.is_file():
            raise FileNotFoundError(f"Missing prior_full for {case_id}: {prior_path}")
        if case_id not in lung_bbox_dict:
            raise KeyError(f"{case_id} not found in {bbox_path}")

        _, image_array, origin, spacing, new_spacing, direction, old_size = load_itk_image_with_sampling(
            str(raw_image_path), spacing=[0.8, 0.8, 0.8], islabel=True
        )
        pred_array, _, _, _ = load_itk_image_new(str(pred_file))
        pred_full = restore_image_via_box_from_json(image_array.shape, pred_array, lung_bbox_dict[case_id])

        prior_array, _, _, _ = load_itk_image_new(str(prior_path))
        if prior_array.shape != pred_full.shape:
            raise ValueError(f"prior_full shape {prior_array.shape} != restored prediction shape {pred_full.shape} for {case_id}")

        prior_exclusion = prior_array.copy()
        prior_exclusion[np.isin(prior_exclusion, [4, 5, 6, 7, 8, 9, 15, 16, 21, 22, 23, 24, 25])] = 0
        prior_exclusion[prior_exclusion > 0] = 1

        save_w_post_del_total(
            pred_full,
            prior_exclusion,
            str(w_post_del_total_dir / pred_file.name),
            origin,
            new_spacing,
            spacing,
            direction,
            old_size,
            islabel=True,
        )
        print(f"[lna-seg] postprocessed {case_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LNA-Seg nnUNetv1 inference and postprocessing.")
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder containing original CT .nii.gz files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--work_dir",
        default=str(DEFAULT_WORK_DIR),
        help=f"Preprocess output folder from stage 1. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Folder containing nnUNet trained models. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--debug_keep_intermediate",
        action="store_true",
        help="Keep raw nnUNet_output for debugging instead of using a temporary folder.",
    )
    parser.add_argument("--cuda_visible_devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")

    parser.add_argument("--task_name", default=DEFAULT_TASK_NAME, help="nnUNet task ID or task name.")
    parser.add_argument("--trainer_class_name", default=None)
    parser.add_argument("--cascade_trainer_class_name", default=None)
    parser.add_argument("--model", default="3d_fullres", choices=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"])
    parser.add_argument("--plans_identifier", default=None)
    parser.add_argument("--folds", nargs="+", default=["None"])
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--lowres_segmentations", default="None")
    parser.add_argument("--part_id", type=int, default=0)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--num_threads_preprocessing", type=int, default=6)
    parser.add_argument("--num_threads_nifti_save", type=int, default=2)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--all_in_gpu", default="None")
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--checkpoint_name", default="model_best")
    parser.add_argument("--disable_mixed_precision", action="store_true")
    parser.add_argument("--skip_prediction", action="store_true", help="Only run postprocessing from an existing nnUNet_output folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    preprocess_dir = resolve_path(args.work_dir)
    model_dir = resolve_path(args.model_dir)
    nnunet_input_dir = preprocess_dir / "nnUNet_input"

    if not nnunet_input_dir.is_dir():
        raise FileNotFoundError(f"Missing nnUNet input folder: {nnunet_input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_prediction:
        nnunet_output_dir = output_dir / "nnUNet_output"
        postprocess_predictions(raw_input_dir, preprocess_dir, nnunet_output_dir, output_dir)
    elif args.debug_keep_intermediate:
        nnunet_output_dir = output_dir / "nnUNet_output"
        nnunet_output_dir.mkdir(parents=True, exist_ok=True)
        run_nnunet_prediction(args, nnunet_input_dir, nnunet_output_dir, model_dir)
        postprocess_predictions(raw_input_dir, preprocess_dir, nnunet_output_dir, output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="nnunet_output_", dir=str(output_dir)) as tmp:
            nnunet_output_dir = Path(tmp)
            run_nnunet_prediction(args, nnunet_input_dir, nnunet_output_dir, model_dir)
            postprocess_predictions(raw_input_dir, preprocess_dir, nnunet_output_dir, output_dir)
    print(f"[lna-seg] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
