import SimpleITK as sitk
import numpy as np
import os
from monai.inferers import sliding_window_inference
from monai.transforms import (
    KeepLargestConnectedComponent,
    ToNumpy,
    AsDiscrete,
    CastToType,
    AddChannel,
    SqueezeDim,
    ToTensor,
    EnsureChannelFirst,
)
import torch
from scipy import ndimage

class InnerTransform(object):
    def __init__(self):
        self.ToNumpy = ToNumpy()
        self.AsDiscrete = AsDiscrete(threshold=0.5)
        self.ArgMax = AsDiscrete(argmax=True)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
        self.EnsureChannelFirst = EnsureChannelFirst()
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.AddChannel = AddChannel()
        self.SqueezeDim = SqueezeDim()
        self.ToTensor = ToTensor(dtype=torch.float32)

InnerTransformer = InnerTransform()

def save_itk(image, filename, origin, spacing, direction):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)

def save_itk_with_backsampling(image, filename, origin, spacing, old_spacing, direction, old_size, islabel=True):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(old_spacing) != tuple:
        if type(old_spacing) == list:
            old_spacing = tuple(reversed(old_spacing))
        else:
            old_spacing = tuple(reversed(old_spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    if type(old_size) != tuple:
        if type(old_size) == list:
            old_size = tuple(reversed(old_size))
        else:
            old_size = tuple(reversed(old_size.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    new_image_sitk, new_spacing_refine = ImageResample_to_newSize(itkimage, newSize=old_size, newSpacing=old_spacing, is_label=islabel)
    sitk.WriteImage(new_image_sitk, filename, True)

def multi_label_connected_components_delate_small(numpyImage_Seg_Class25, spacing, min_volume=65):

    unique_labels = np.unique(numpyImage_Seg_Class25)
    unique_labels = unique_labels[unique_labels != 0]
    filtered_image = np.zeros_like(numpyImage_Seg_Class25)
    kept_components = 0
    del_components = 0
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    for label in unique_labels:
        mask = (numpyImage_Seg_Class25 == label).astype(np.uint8)
        labeled, num_features = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        component_sizes = np.bincount(labeled.ravel())
        for component_id in range(1, num_features + 1):
            if component_sizes[component_id] * voxel_volume >= min_volume:
                filtered_image[labeled == component_id] = label
                kept_components += 1
            else:
                del_components += 1
    return filtered_image, kept_components, del_components

def save_itk_with_backsampling_with_ConnectedComponent(image, image_prior, filename, filename_post, filename_post_delTotal, origin, spacing, old_spacing, direction, old_size, islabel=True):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(old_spacing) != tuple:
        if type(old_spacing) == list:
            old_spacing = tuple(reversed(old_spacing))
        else:
            old_spacing = tuple(reversed(old_spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    if type(old_size) != tuple:
        if type(old_size) == list:
            old_size = tuple(reversed(old_size))
        else:
            old_size = tuple(reversed(old_size.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    new_image_sitk, new_spacing_refine = ImageResample_to_newSize(itkimage, newSize=old_size, newSpacing=old_spacing, is_label=islabel)
    sitk.WriteImage(new_image_sitk, filename, True)

    seg_array_delSmall, seg_kept_components, seg_del_components = multi_label_connected_components_delate_small(
        sitk.GetArrayFromImage(new_image_sitk), new_image_sitk.GetSpacing())
    new_image_sitk_post = sitk.GetImageFromArray(seg_array_delSmall, isVector=False)
    new_image_sitk_post.SetSpacing(old_spacing)
    new_image_sitk_post.SetOrigin(origin)
    new_image_sitk_post.SetDirection(direction)
    sitk.WriteImage(new_image_sitk_post, filename_post, True)

    itkimage_prior = sitk.GetImageFromArray(image_prior, isVector=False)
    itkimage_prior.SetSpacing(spacing)
    itkimage_prior.SetOrigin(origin)
    itkimage_prior.SetDirection(direction)
    new_image_prior_sitk, new_spacing_prior_refine = ImageResample_to_newSize(itkimage_prior, newSize=old_size, newSpacing=old_spacing, is_label=islabel)
    new_image_prior_sitk_array = sitk.GetArrayFromImage(new_image_prior_sitk)
    new_image_sitk_post_array = sitk.GetArrayFromImage(new_image_sitk_post)
    new_image_sitk_post_array[new_image_prior_sitk_array == 1] = 0
    new_image_sitk_post_array_delSmall, seg_kept_components, seg_del_components = multi_label_connected_components_delate_small(
        new_image_sitk_post_array, old_spacing)
    itkimage = sitk.GetImageFromArray(new_image_sitk_post_array_delSmall, isVector=False)
    itkimage.SetSpacing(old_spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename_post_delTotal, True)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyDirection

def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
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

    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine, size

def ImageResample_to_newSize(sitk_image, newSize, newSpacing, is_label=False):

    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_size = np.array(newSize, float)
    new_spacing = np.array(newSpacing, float)
    factor = size/new_size
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

    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine

def load_itk_image_with_sampling(filename, spacing=[0.8, 0.8, 0.8], islabel=False):
    itkimage = sitk.ReadImage(filename)
    new_image_sitk, new_spacing_refine, old_size = ImageResample(itkimage, new_spacing=spacing, is_label=islabel)
    numpyImage = sitk.GetArrayFromImage(new_image_sitk)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return new_image_sitk, numpyImage, numpyOrigin, numpySpacing, list(reversed(new_spacing_refine)), numpyDirection, list(reversed(old_size))

def crop_image_via_box(image, box):
    image_shape = image.shape
    crop_coords = []
    for dim in range(3):
        start = max(0, box[dim][0])
        end = min(image_shape[dim], box[dim][1])
        if start >= end:
            return np.array([])
        crop_coords.append((start, end))
    return image[crop_coords[0][0]:crop_coords[0][1],
                crop_coords[1][0]:crop_coords[1][1],
                crop_coords[2][0]:crop_coords[2][1]]

def restore_image_via_box(origin_shape, image, box):
    origin_image = np.zeros(shape=origin_shape, dtype=np.uint8)
    origin_image[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]] = image
    return origin_image

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

