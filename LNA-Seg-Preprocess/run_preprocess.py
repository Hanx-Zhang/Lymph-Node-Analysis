import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INITIAL_CWD = Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_INPUT_DIR = REPO_ROOT / "LNA-Inputs" / "case_data_nii"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "LNA-Outputs" / "01_preprocess"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / "_work"

from anatomical_prior_models.airway_model import AirwayExtractionModel
from anatomical_prior_models.azygos_model import AzygosModel
from anatomical_prior_models.lungmask_model import LungMaskExtractionModel
from anatomical_prior_models.utils import (
    ImageResample,
    InnerTransformer,
    crop_image_via_box,
    load_itk_image,
    mkdir,
    restore_image_via_box,
    save_itk,
)
from totalseg_api import totalsegmentator


TOTAL_ROI_SUBSET = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "thyroid_gland",
    "esophagus",
    "aorta",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "superior_vena_cava",
    "inferior_vena_cava",
    "pulmonary_vein",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = INITIAL_CWD / path
    return path.resolve()


def case_id_from_nii(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def list_nii_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.name.endswith(".nii.gz")])


def load_itk_image_new(filename: Path):
    itkimage = sitk.ReadImage(str(filename))
    numpy_image = sitk.GetArrayFromImage(itkimage)
    numpy_origin = list(reversed(itkimage.GetOrigin()))
    numpy_spacing = list(reversed(itkimage.GetSpacing()))
    numpy_direction = list(reversed(itkimage.GetDirection()))
    return numpy_image, numpy_origin, numpy_spacing, numpy_direction


def save_itk_new(image, filename: Path, origin, spacing, direction):
    save_itk(image, str(filename), origin, spacing, direction)


def locate_airway_boundingbox(image: np.ndarray) -> np.ndarray:
    xx, yy, zz = np.where(image)
    if len(xx) == 0:
        raise ValueError("Airway mask is empty; cannot crop nnUNet input.")
    bbox = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    margin = 20
    bbox = np.vstack(
        [
            np.max([[0, 0, 0], bbox[:, 0] - margin], 0),
            np.min([np.array(image.shape), bbox[:, 1] + margin], axis=0).T,
        ]
    ).T
    return bbox


def locate_azygos_boundingbox(image: np.ndarray, margin: int = 10, min_size: int = 128) -> np.ndarray:
    xx, yy, zz = np.where(image)
    if len(xx) == 0:
        raise ValueError("Input mask does not contain any non-zero region.")

    bbox_min = np.array([np.min(xx), np.min(yy), np.min(zz)]) - margin
    bbox_max = np.array([np.max(xx), np.max(yy), np.max(zz)]) + margin
    bbox_min = np.maximum(bbox_min, 0)
    bbox_max = np.minimum(bbox_max, np.array(image.shape))

    for axis in range(3):
        size = bbox_max[axis] - bbox_min[axis] + 1
        if size < min_size:
            extra = min_size - size
            before = extra // 2
            after = extra - before
            bbox_min[axis] = max(bbox_min[axis] - before, 0)
            bbox_max[axis] = min(bbox_max[axis] + after + 1, image.shape[axis])

    return np.stack([bbox_min, bbox_max], axis=1).astype(int)


def intersect_bounding_boxes_min128(box1, box2, min_size: int = 128, image_shape=None):
    intersection_min = np.maximum(box1[:, 0], box2[:, 0])
    intersection_max = np.minimum(box1[:, 1], box2[:, 1])
    if np.any(intersection_max < intersection_min):
        return None

    size = intersection_max - intersection_min + 1
    for axis in range(3):
        if size[axis] < min_size:
            extra = min_size - size[axis]
            before = extra // 2
            after = extra - before
            intersection_min[axis] -= before
            intersection_max[axis] += after + 1

    if image_shape is not None:
        for axis in range(3):
            intersection_min[axis] = max(0, intersection_min[axis])
            intersection_max[axis] = min(image_shape[axis] - 1, intersection_max[axis])
            current_size = intersection_max[axis] - intersection_min[axis] + 1
            if current_size < min_size:
                needed = min_size - current_size
                shift_before = min(intersection_min[axis], needed // 2)
                shift_after = min(image_shape[axis] - 1 - intersection_max[axis], needed - shift_before)
                intersection_min[axis] -= shift_before
                intersection_max[axis] += shift_after

    return np.stack([intersection_min, intersection_max], axis=1).astype(int)


def largest_component_per_label(mask_3d: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask_3d)
    for label_val in np.unique(mask_3d):
        if label_val == 0:
            continue
        binary_mask = (mask_3d == label_val).astype(int)
        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            continue
        largest_component = 0
        max_size = 0
        for component_id in range(1, num_features + 1):
            component_size = np.sum(labeled_array == component_id)
            if component_size > max_size:
                max_size = component_size
                largest_component = component_id
        output[labeled_array == largest_component] = label_val
    return output


def compute_crop_origin(origin, spacing, direction, bbox):
    affine = np.eye(4)
    affine[0, :3] = np.asarray(direction[:3]) * spacing[0]
    affine[1, :3] = np.asarray(direction[3:6]) * spacing[1]
    affine[2, :3] = np.asarray(direction[6:9]) * spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    origin_point = [[bbox[0, 0], bbox[1, 0], bbox[2, 0], 1]]
    origin_world = np.dot(affine, np.transpose(origin_point))
    return np.transpose(origin_world[0:3, 0]).tolist()


def write_onehot_prior(prior_crop_path: Path, output_dir: Path, nnunet_input_dir: Path):
    image = sitk.ReadImage(str(prior_crop_path))
    array = sitk.GetArrayFromImage(image)
    case_prefix = prior_crop_path.name.replace("_0001.nii.gz", "_")

    channel_defs = {
        1: [3],
        2: [10],
        3: [12, 13, 14],
        4: [17, 18],
        5: [26],
    }
    for channel, labels in channel_defs.items():
        out_array = np.isin(array, labels).astype(np.uint8)
        out_image = sitk.GetImageFromArray(out_array, isVector=False)
        out_image.CopyInformation(image)
        filename = f"{case_prefix}{channel:04d}.nii.gz"
        output_path = output_dir / filename
        sitk.WriteImage(out_image, str(output_path), True)
        shutil.copy2(output_path, nnunet_input_dir / filename)


def build_prior(
    main_array,
    heart_array,
    nodule_array,
    airway_array,
    azygos_array,
) -> np.ndarray:
    prior = np.zeros_like(main_array, dtype=np.uint8)
    prior[azygos_array == 1] = 26

    prior[heart_array == 1] = 21
    prior[heart_array == 2] = 22
    prior[heart_array == 3] = 23
    prior[heart_array == 4] = 24
    prior[heart_array == 5] = 25

    prior[main_array == 10] = 5
    prior[main_array == 11] = 6
    prior[main_array == 12] = 7
    prior[main_array == 13] = 8
    prior[main_array == 14] = 9
    prior[main_array == 15] = 10
    prior[main_array == 52] = 11
    prior[main_array == 54] = 12
    prior[main_array == 55] = 13
    prior[main_array == 56] = 14
    prior[main_array == 57] = 15
    prior[main_array == 58] = 16
    prior[main_array == 59] = 17
    prior[main_array == 60] = 18
    prior[main_array == 62] = 19
    prior[main_array == 63] = 20

    prior[heart_array == 7] = 1
    prior[main_array == 53] = 2
    prior[airway_array == 1] = 3
    prior[nodule_array == 2] = 4
    return prior


def process_case(case_path: Path, output_dir: Path, total_dir: Path, debug_keep_intermediate: bool):
    case_id = case_id_from_nii(case_path)
    start_time = time.time()
    print(f"[preprocess] start {case_id}")

    main_path = total_dir / f"{case_id}_main.nii.gz"
    heart_path = total_dir / f"{case_id}_heartchambers.nii.gz"
    nodule_path = total_dir / f"{case_id}_nodule.nii.gz"
    total_dir.mkdir(parents=True, exist_ok=True)

    totalsegmentator(str(case_path), str(main_path), ml=True, roi_subset_robust=TOTAL_ROI_SUBSET, remove_small_blobs=True)
    totalsegmentator(str(case_path), str(heart_path), ml=True, task="heartchambers_highres", remove_small_blobs=True)
    totalsegmentator(str(case_path), str(nodule_path), ml=True, task="lung_nodules", remove_small_blobs=True)

    image_sitk = sitk.ReadImage(str(case_path))
    image_array, origin, _, direction = load_itk_image(str(case_path))
    main_sitk = sitk.ReadImage(str(main_path))
    heart_sitk = sitk.ReadImage(str(heart_path))
    nodule_sitk = sitk.ReadImage(str(nodule_path))

    new_image_sitk, new_spacing_refine, _ = ImageResample(image_sitk, new_spacing=[0.8, 0.8, 0.8], is_label=False)
    new_image_array = sitk.GetArrayFromImage(new_image_sitk)
    main_resampled_sitk, _, _ = ImageResample(main_sitk, new_spacing=[0.8, 0.8, 0.8], is_label=True)
    heart_resampled_sitk, _, _ = ImageResample(heart_sitk, new_spacing=[0.8, 0.8, 0.8], is_label=True)
    nodule_resampled_sitk, _, _ = ImageResample(nodule_sitk, new_spacing=[0.8, 0.8, 0.8], is_label=True)

    main_array = sitk.GetArrayFromImage(main_resampled_sitk)
    heart_array = sitk.GetArrayFromImage(heart_resampled_sitk)
    nodule_array = sitk.GetArrayFromImage(nodule_resampled_sitk)

    lobe_extractor = LungMaskExtractionModel()
    lobe = lobe_extractor.predict(new_image_sitk)
    lung_voxels = np.where(lobe)
    if len(lung_voxels[0]) == 0:
        raise ValueError(f"Lung mask is empty for {case_id}")
    lung_bbox = np.array(
        [
            [np.min(lung_voxels[0]), np.max(lung_voxels[0])],
            [np.min(lung_voxels[1]), np.max(lung_voxels[1])],
            [np.min(lung_voxels[2]), np.max(lung_voxels[2])],
        ]
    )
    lung_bbox = np.vstack(
        [
            np.max([[0, 0, 0], lung_bbox[:, 0]], 0),
            np.min([np.array(lobe.shape), lung_bbox[:, 1]], axis=0).T,
        ]
    ).T
    lung_bbox[0, 1] = min(lung_bbox[0, 1] + 20, new_image_array.shape[0])

    image_lung_crop = crop_image_via_box(new_image_array, lung_bbox)
    airway_extractor = AirwayExtractionModel()
    airway_crop = airway_extractor.predict(image_lung_crop)
    airway_lcc = InnerTransformer.KeepLargestConnectedComponent(airway_crop)
    airway_post = InnerTransformer.ToNumpy(airway_lcc)
    airway_post = InnerTransformer.CastToNumpyUINT8(airway_post[0, ...])
    torch.cuda.empty_cache()
    airway_full = restore_image_via_box(new_image_array.shape, airway_post, lung_bbox)

    aorta_svc = np.zeros_like(main_array)
    aorta_svc[main_array == 52] = 1
    aorta_svc[main_array == 62] = 1
    esophagus = np.zeros_like(main_array)
    esophagus[main_array == 15] = 1
    azygos_box = locate_azygos_boundingbox(aorta_svc, margin=30, min_size=128)
    esophagus_box = locate_azygos_boundingbox(esophagus, margin=30, min_size=128)
    azygos_intersection = intersect_bounding_boxes_min128(
        azygos_box, esophagus_box, min_size=128, image_shape=main_array.shape
    )
    if azygos_intersection is None:
        raise ValueError(f"Azygos crop boxes do not intersect for {case_id}")
    azygos_image_crop = crop_image_via_box(new_image_array, azygos_intersection)
    azygos_extractor = AzygosModel()
    azygos_crop = azygos_extractor.predict(azygos_image_crop)
    azygos_full = restore_image_via_box(new_image_array.shape, azygos_crop, azygos_intersection)
    azygos_post = largest_component_per_label(azygos_full)

    prior_full = build_prior(main_array, heart_array, nodule_array, airway_full, azygos_post)

    airway_wo_lobe = airway_full.copy()
    airway_wo_lobe[lobe > 0] = 0
    airway_bbox = locate_airway_boundingbox(airway_wo_lobe)
    crop_bbox = airway_bbox.copy()
    crop_bbox[0, 0] = max(0, crop_bbox[0, 0] - 20)
    crop_bbox[0, 1] = lung_bbox[0, 1]
    crop_bbox[1, 0] = min(max(0, lung_bbox[1, 0] + 20), new_image_array.shape[1])
    crop_bbox[1, 1] = max(0, lung_bbox[1, 1] - 20)
    crop_bbox[2, 0] = max(0, crop_bbox[2, 0])
    crop_bbox[2, 1] = min(new_image_array.shape[2], crop_bbox[2, 1])

    image_crop = crop_image_via_box(new_image_array, crop_bbox)
    prior_crop = crop_image_via_box(prior_full, crop_bbox)
    crop_origin = compute_crop_origin(origin, new_spacing_refine, direction, crop_bbox)

    prior_full_dir = output_dir / "prior_full"
    prior_crop_dir = output_dir / "prior_crop"
    onehot_dir = output_dir / "prior_crop_onehot_5structure"
    nnunet_input_dir = output_dir / "nnUNet_input"
    for folder in [prior_full_dir, prior_crop_dir, onehot_dir, nnunet_input_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    prior_full_path = prior_full_dir / f"{case_id}.nii.gz"
    prior_crop_path = prior_crop_dir / f"{case_id}_0001.nii.gz"
    ct_crop_path = nnunet_input_dir / f"{case_id}_0000.nii.gz"
    save_itk(prior_full, str(prior_full_path), origin, tuple(new_spacing_refine), direction)
    save_itk(prior_crop, str(prior_crop_path), crop_origin, tuple(new_spacing_refine), direction)
    save_itk_new(image_crop, ct_crop_path, crop_origin, tuple(new_spacing_refine), direction)
    write_onehot_prior(prior_crop_path, onehot_dir, nnunet_input_dir)

    if debug_keep_intermediate:
        debug_dir = output_dir / "debug"
        for folder in ["ct_resampled", "lung_lobe", "airway", "azygos"]:
            (debug_dir / folder).mkdir(parents=True, exist_ok=True)
        save_itk(new_image_array, str(debug_dir / "ct_resampled" / f"{case_id}.nii.gz"), origin, tuple(new_spacing_refine), direction)
        save_itk(lobe, str(debug_dir / "lung_lobe" / f"{case_id}.nii.gz"), origin, tuple(new_spacing_refine), direction)
        save_itk(airway_full, str(debug_dir / "airway" / f"{case_id}.nii.gz"), origin, tuple(new_spacing_refine), direction)
        save_itk(azygos_post, str(debug_dir / "azygos" / f"{case_id}.nii.gz"), origin, tuple(new_spacing_refine), direction)

    bbox_flat = np.reshape(crop_bbox, newshape=(6,))
    print(f"[preprocess] done {case_id} in {time.time() - start_time:.1f}s")
    return case_id, [int(x) for x in bbox_flat]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LNA-Seg nnUNet input with TotalSegmentator priors.")
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
        help=f"Temporary work folder. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument("--debug_keep_intermediate", action="store_true", help="Keep debug intermediate masks.")
    parser.add_argument("--cuda_visible_devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    work_dir = resolve_path(args.work_dir) if args.work_dir else output_dir / "_work"

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    os.chdir(SCRIPT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    case_files = list_nii_files(input_dir)
    if not case_files:
        raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cases": [],
        "notes": "Model weights and data are not part of the open-source repository.",
    }
    bbox_dict = {}

    if args.debug_keep_intermediate:
        total_dir = output_dir / "debug" / "totalseg_raw"
        total_dir.mkdir(parents=True, exist_ok=True)
        for case_file in case_files:
            case_id, bbox = process_case(case_file, output_dir, total_dir, args.debug_keep_intermediate)
            bbox_dict[case_id] = bbox
            manifest["cases"].append(case_id)
    else:
        with tempfile.TemporaryDirectory(dir=str(work_dir)) as tmp:
            total_dir = Path(tmp) / "totalseg_raw"
            for case_file in case_files:
                case_id, bbox = process_case(case_file, output_dir, total_dir, args.debug_keep_intermediate)
                bbox_dict[case_id] = bbox
                manifest["cases"].append(case_id)

    with (output_dir / "lung_bbox_dict.json").open("w", encoding="utf-8") as f:
        json.dump(bbox_dict, f, indent=2)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[preprocess] wrote {len(bbox_dict)} cases to {output_dir}")


if __name__ == "__main__":
    main()
