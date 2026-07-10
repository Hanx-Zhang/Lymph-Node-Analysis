import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def multi_label_connected_components_delate_small(seg_array, spacing, min_volume=65):
    unique_labels = np.unique(seg_array)
    unique_labels = unique_labels[unique_labels != 0]
    filtered_image = np.zeros_like(seg_array)
    kept_components = 0
    deleted_components = 0
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    for label in unique_labels:
        mask = (seg_array == label).astype(np.uint8)
        labeled, num_features = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        component_sizes = np.bincount(labeled.ravel())

        for component_id in range(1, num_features + 1):
            if component_sizes[component_id] * voxel_volume >= min_volume:
                filtered_image[labeled == component_id] = label
                kept_components += 1
            else:
                deleted_components += 1

    return filtered_image, kept_components, deleted_components


def ImageResample(sitk_image, new_spacing=(1.0, 1.0, 1.0), is_label=False):
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

    new_image = resample.Execute(sitk_image)
    return new_image, new_spacing_refine, size


def ImageResample_to_newSize(sitk_image, newSize, newSpacing, is_label=False):
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_size = np.array(newSize, float)
    new_spacing = np.array(newSpacing, float)
    factor = size / new_size
    new_spacing_refine = spacing * factor
    new_size = new_size.astype(int)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size.tolist())
    resample.SetOutputSpacing(new_spacing)

    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)

    new_image = resample.Execute(sitk_image)
    return new_image, new_spacing_refine


def load_itk_image_with_sampling(filename, spacing=(0.8, 0.8, 0.8), islabel=False):
    itk_image = sitk.ReadImage(filename)
    new_image_sitk, new_spacing_refine, old_size = ImageResample(
        itk_image, new_spacing=spacing, is_label=islabel
    )
    image_array = sitk.GetArrayFromImage(new_image_sitk)
    origin = list(reversed(itk_image.GetOrigin()))
    original_spacing = list(reversed(itk_image.GetSpacing()))
    direction = list(reversed(itk_image.GetDirection()))
    return (
        new_image_sitk,
        image_array,
        origin,
        original_spacing,
        list(reversed(new_spacing_refine)),
        direction,
        list(reversed(old_size)),
    )
