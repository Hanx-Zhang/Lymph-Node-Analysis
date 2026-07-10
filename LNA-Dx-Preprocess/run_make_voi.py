import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, label


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INITIAL_CWD = Path.cwd()

DEFAULT_INPUT_DIR = REPO_ROOT / "LNA-Inputs" / "case_data_nii"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "LNA-Outputs" / "03_voi_for_lna_dx"
DEFAULT_WORK_DIR = REPO_ROOT / "LNA-Outputs"
DEFAULT_CASE_TABLE = REPO_ROOT / "LNA-Inputs" / "case_info" / "case_table_local.csv"


def resolve_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = INITIAL_CWD / path
    return path.resolve()


def case_id_from_nii_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def load_itk_image(filename: Path):
    itkimage = sitk.ReadImage(str(filename))
    numpy_image = sitk.GetArrayFromImage(itkimage)
    numpy_origin = list(reversed(itkimage.GetOrigin()))
    numpy_spacing = list(reversed(itkimage.GetSpacing()))
    numpy_direction = list(reversed(itkimage.GetDirection()))
    return numpy_image, numpy_origin, numpy_spacing, numpy_direction


def to_sitk_tuple(values):
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(reversed(values))
    return tuple(reversed(values.tolist()))


def save_itk(image, filename: Path, origin, spacing, direction):
    filename.parent.mkdir(parents=True, exist_ok=True)
    origin = to_sitk_tuple(origin)
    spacing = to_sitk_tuple(spacing)
    direction = to_sitk_tuple(direction)
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, str(filename), True)


def image_resample(sitk_image, new_spacing=(1.0, 1.0, 1.0), is_label=False):
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(round(s, 7)) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)
    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(sitk_image), new_spacing_refine, size


def load_itk_image_with_sampling(filename: Path, spacing=(0.8, 0.8, 0.8), islabel=False):
    itkimage = sitk.ReadImage(str(filename))
    new_image_sitk, new_spacing_refine, old_size = image_resample(itkimage, new_spacing=spacing, is_label=islabel)
    numpy_image = sitk.GetArrayFromImage(new_image_sitk)
    numpy_origin = list(reversed(itkimage.GetOrigin()))
    numpy_spacing = list(reversed(itkimage.GetSpacing()))
    numpy_direction = list(reversed(itkimage.GetDirection()))
    return new_image_sitk, numpy_image, numpy_origin, numpy_spacing, list(reversed(new_spacing_refine)), numpy_direction, list(reversed(old_size))


def image_resample_to_new_size(sitk_image, new_size, new_spacing, is_label=False):
    new_size = np.array(new_size, float).astype(int)
    new_spacing = np.array(new_spacing, float)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size.tolist())
    resample.SetOutputSpacing(new_spacing.tolist())
    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(sitk_image)


def from_itk_with_backsampling(image, origin, spacing, old_spacing, direction, old_size, islabel=True):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(to_sitk_tuple(spacing))
    itkimage.SetOrigin(to_sitk_tuple(origin))
    itkimage.SetDirection(to_sitk_tuple(direction))
    new_image_sitk = image_resample_to_new_size(
        itkimage,
        new_size=to_sitk_tuple(old_size),
        new_spacing=to_sitk_tuple(old_spacing),
        is_label=islabel,
    )
    return sitk.GetArrayFromImage(new_image_sitk)


def locate_boundingbox(image, margin):
    xx, yy, zz = np.where(image)
    if len(xx) == 0 or len(yy) == 0 or len(zz) == 0:
        return None
    bbox = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    bbox = np.vstack(
        [
            np.max([[0, 0, 0], bbox[:, 0] - margin], 0),
            np.min([np.array(image.shape), bbox[:, 1] + margin], axis=0).T,
        ]
    ).T
    return bbox


def get_bbox_intersection(bbox1, bbox2):
    intersection = np.array(
        [
            [max(bbox1[0, 0], bbox2[0, 0]), min(bbox1[0, 1], bbox2[0, 1])],
            [max(bbox1[1, 0], bbox2[1, 0]), min(bbox1[1, 1], bbox2[1, 1])],
            [max(bbox1[2, 0], bbox2[2, 0]), min(bbox1[2, 1], bbox2[2, 1])],
        ]
    )
    if np.any(intersection[:, 0] >= intersection[:, 1]):
        return None
    return intersection


def crop_image_via_box(image, box):
    return image[box[0, 0] : box[0, 1], box[1, 0] : box[1, 1], box[2, 0] : box[2, 1]]


def extract_largest_3d_component(binary_mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    if binary_mask.ndim != 3:
        raise ValueError("Input must be a 3D array")
    if np.all(binary_mask == 0):
        return np.zeros_like(binary_mask, dtype=np.uint8)
    labeled_mask, _ = label(binary_mask.astype(bool), structure=np.ones((3, 3, 3)) if connectivity == 1 else None)
    component_sizes = np.bincount(labeled_mask.ravel())[1:]
    largest_label = np.argmax(component_sizes) + 1
    return (labeled_mask == largest_label).astype(np.uint8)


def filter_seg_array_by_lung(seg_array, which_lung):
    seg_array_new = np.zeros_like(seg_array, dtype=seg_array.dtype)
    if which_lung == "right":
        keep_labels = list(range(2, 15)) + [26]
    elif which_lung == "left":
        keep_labels = list(range(5, 8)) + list(range(16, 26)) + [26]
    else:
        keep_labels = list(range(2, 27))
    seg_array_new[np.isin(seg_array, keep_labels)] = seg_array[np.isin(seg_array, keep_labels)]
    return seg_array_new


LEFT_LUNG_PRIOR_LABELS = (5, 6)
RIGHT_LUNG_PRIOR_LABELS = (7, 8, 9)


def parse_side(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in ("left", "l", "2", "lt") or "左" in text:
        return "left"
    if text in ("right", "r", "1", "rt") or "右" in text:
        return "right"
    if text in ("other", "unknown", "unk", "0"):
        return "other"
    return None


def read_case_table(case_table: Optional[Path]) -> Dict[str, Optional[str]]:
    if case_table is None:
        return {}
    if case_table.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(case_table)
    else:
        frame = pd.read_csv(case_table)
    if frame.empty:
        return {}
    columns = {c.lower(): c for c in frame.columns}
    case_col = columns.get("case_id") or columns.get("case") or columns.get("id") or frame.columns[0]
    side_col = columns.get("tumor_side") or columns.get("side")
    if side_col is None:
        if len(frame.columns) < 2:
            return {}
        side_col = frame.columns[1]
    return {str(row[case_col]): parse_side(row[side_col]) for _, row in frame.iterrows()}


def infer_side_from_largest_nodule(seg_array: np.ndarray, prior_array: np.ndarray) -> str:
    if seg_array.shape != prior_array.shape:
        return "other"

    nodule_mask = seg_array == 26
    if not np.any(nodule_mask):
        return "other"

    nodule_box = locate_boundingbox(nodule_mask, margin=40)
    if nodule_box is None:
        return "other"

    local_nodule = crop_image_via_box(nodule_mask, nodule_box)
    local_prior = crop_image_via_box(prior_array, nodule_box)
    structure = np.ones((3, 3, 3), dtype=bool)

    for iterations in (0, 2, 5, 10, 20, 40):
        if iterations == 0:
            region = local_nodule
        else:
            region = binary_dilation(local_nodule, structure=structure, iterations=iterations)

        region_labels = local_prior[region]
        left_count = int(np.count_nonzero(np.isin(region_labels, LEFT_LUNG_PRIOR_LABELS)))
        right_count = int(np.count_nonzero(np.isin(region_labels, RIGHT_LUNG_PRIOR_LABELS)))

        if left_count > right_count:
            return "left"
        if right_count > left_count:
            return "right"
    return "other"


def choose_case_side(case_id: str, case_sides: Dict[str, Optional[str]], seg_array: np.ndarray, prior_array: np.ndarray):
    table_side = case_sides.get(case_id)
    if table_side in ("left", "right", "other"):
        return table_side, "case_table"
    return infer_side_from_largest_nodule(seg_array, prior_array), "auto_nodule_lung"


def write_seg_with_largest_nodule(seg_dir: Path, prior_full_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_files = sorted([p for p in seg_dir.iterdir() if p.name.endswith(".nii.gz")])
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files found in {seg_dir}")

    for seg_file in seg_files:
        prior_file = prior_full_dir / seg_file.name
        if not prior_file.is_file():
            raise FileNotFoundError(f"Missing prior_full for {seg_file.name}: {prior_file}")

        seg_array, origin, spacing, direction = load_itk_image(seg_file)
        seg_sitk = sitk.ReadImage(str(seg_file))
        prior_array, _, prior_spacing, _ = load_itk_image(prior_file)
        prior_back = from_itk_with_backsampling(
            prior_array,
            origin,
            prior_spacing,
            spacing,
            direction,
            tuple(np.array(seg_sitk.GetSize())),
            islabel=True,
        )

        nodule_mask = np.zeros_like(seg_array, dtype=np.uint8)
        nodule_mask[prior_back == 4] = 1
        largest_nodule = extract_largest_3d_component(nodule_mask)
        seg_w_largest = seg_array.copy()
        seg_w_largest[largest_nodule == 1] = 26
        save_itk(seg_w_largest, output_dir / seg_file.name, origin, spacing, direction)
        print(f"[voi] prepared nodule-largest mask {case_id_from_nii_name(seg_file.name)}")

    return output_dir


def make_voi(raw_input_dir: Path, seg_with_nodule_dir: Path, prior_full_dir: Path, output_dir: Path, case_sides: Dict[str, Optional[str]]):
    img_voi_dir = output_dir / "img_VOI"
    seg_voi_dir = output_dir / "seg_VOI"
    img_voi_dir.mkdir(parents=True, exist_ok=True)
    seg_voi_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for seg_file in sorted([p for p in seg_with_nodule_dir.iterdir() if p.name.endswith(".nii.gz")]):
        case_id = case_id_from_nii_name(seg_file.name)
        raw_image = raw_input_dir / seg_file.name
        prior_file = prior_full_dir / seg_file.name
        if not raw_image.is_file():
            raise FileNotFoundError(f"Missing raw image for {case_id}: {raw_image}")
        if not prior_file.is_file():
            raise FileNotFoundError(f"Missing prior_full for {case_id}: {prior_file}")

        _, seg_array, origin, _, new_spacing, direction, _ = load_itk_image_with_sampling(seg_file, spacing=[0.8, 0.8, 0.8], islabel=True)
        _, img_array, _, _, _, _, _ = load_itk_image_with_sampling(raw_image, spacing=[0.8, 0.8, 0.8], islabel=False)
        _, prior_array, _, _, _, _, _ = load_itk_image_with_sampling(prior_file, spacing=[0.8, 0.8, 0.8], islabel=True)

        side, side_source = choose_case_side(case_id, case_sides, seg_array, prior_array)
        seg_array_full = seg_array.copy()
        seg_array_for_bbox = filter_seg_array_by_lung(seg_array_full, side)
        lung_mask = np.isin(prior_array, [5, 6, 7, 8, 9]).astype(np.uint8)
        if lung_mask.shape != seg_array.shape:
            lung_mask = np.ones_like(seg_array, dtype=np.uint8)

        bbox_lung = locate_boundingbox(lung_mask, margin=5)
        bbox_ln = locate_boundingbox(seg_array_for_bbox, margin=10)
        if bbox_lung is None or bbox_ln is None:
            skipped.append({"case_id": case_id, "reason": "empty lung or LN/nodule mask"})
            continue
        intersection = get_bbox_intersection(bbox_lung, bbox_ln)
        if intersection is None:
            intersection = bbox_ln

        image_crop = crop_image_via_box(img_array, intersection)
        seg_crop = crop_image_via_box(seg_array_full, intersection)
        min_index = np.array([intersection[0, 0], intersection[1, 0], intersection[2, 0]])
        offset_physical = np.array(direction).reshape(3, 3) @ (np.array(new_spacing) * min_index)
        new_origin = np.array(origin) + offset_physical

        save_itk(image_crop, img_voi_dir / seg_file.name, new_origin, new_spacing, direction)
        save_itk(seg_crop, seg_voi_dir / seg_file.name, new_origin, new_spacing, direction)
        print(f"[voi] wrote VOI {case_id} side={side} source={side_source}")

    if skipped:
        skip_path = output_dir / "skipped_cases.csv"
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
        print(f"[voi] skipped {len(skipped)} cases; see {skip_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LNA-Dx VOI images and masks from LNA-Seg outputs.")
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
        help=f"Pipeline output root containing 01_preprocess and 02_lna_seg. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--case_table",
        default=None,
        help=f"Optional CSV/XLSX with columns case_id,tumor_side. If omitted, auto-use {DEFAULT_CASE_TABLE} when it exists.",
    )
    parser.add_argument("--preprocess_dir", default=None, help="Optional explicit preprocess output folder.")
    parser.add_argument("--seg_dir", default=None, help="Optional explicit LNA-Seg output folder.")
    parser.add_argument("--debug_keep_intermediate", action="store_true", help="Keep intermediate nodule-largest masks.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    work_dir = resolve_path(args.work_dir)
    preprocess_dir = resolve_path(args.preprocess_dir) if args.preprocess_dir else work_dir / "01_preprocess"
    seg_output_dir = resolve_path(args.seg_dir) if args.seg_dir else work_dir / "02_lna_seg"
    if args.case_table:
        case_table = resolve_path(args.case_table)
        if not case_table.is_file():
            raise FileNotFoundError(f"Missing case table: {case_table}")
    elif DEFAULT_CASE_TABLE.is_file():
        case_table = DEFAULT_CASE_TABLE
        print(f"[voi] using local case table: {case_table}")
    else:
        case_table = None

    prior_full_dir = preprocess_dir / "prior_full"
    seg_dir = seg_output_dir / "wPost_delTotal"
    if not prior_full_dir.is_dir():
        raise FileNotFoundError(f"Missing prior_full folder: {prior_full_dir}")
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Missing LNA-Seg wPost_delTotal folder: {seg_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_sides = read_case_table(case_table)
    if args.debug_keep_intermediate:
        seg_with_nodule_dir = output_dir / "intermediate_seg_with_nodule_largest"
        write_seg_with_largest_nodule(seg_dir, prior_full_dir, seg_with_nodule_dir)
        make_voi(raw_input_dir, seg_with_nodule_dir, prior_full_dir, output_dir, case_sides)
    else:
        with tempfile.TemporaryDirectory(prefix="seg_with_nodule_largest_", dir=str(output_dir)) as tmp:
            seg_with_nodule_dir = write_seg_with_largest_nodule(seg_dir, prior_full_dir, Path(tmp))
            make_voi(raw_input_dir, seg_with_nodule_dir, prior_full_dir, output_dir, case_sides)
    print(f"[voi] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
