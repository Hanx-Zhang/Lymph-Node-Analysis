# Lymph Node Analysis

This repository contains the GitHub-ready code release for the Lymph Node
Analysis (LNA) pipeline described in the manuscript "Anatomically Grounded, Scalable, and Interpretable
CT Analysis of Thoracic Lymph Node Metastasis in Lung Cancer". The pipeline links four
stages:

1. `LNA-Seg-Preprocess`: prepare CT volumes and anatomical priors.
2. `LNA-Seg`: run lymph-node segmentation with nnU-Net.
3. `LNA-Dx-Preprocess`: generate VOIs for diagnosis.
4. `LNA-Dx`: train or run the diagnosis model.

## Repository Layout

```text
Lymph-Node-Analysis/
  README.md
  requirements-totalseg.txt
  requirements-pytorch.txt
  LNA-Inputs/
    case_data_nii/                 # local test CT inputs; empty in GitHub
    case_info/
      case_table_template.csv       # public template only
  LNA-Seg-Preprocess/               # Stage 1
  LNA-Seg/                          # Stage 2, includes vendored nnU-Net code
  LNA-Dx-Preprocess/                # Stage 3
  LNA-Dx/                           # Stage 4
```

`LNA-Inputs` is only a convenient input location for local testing. For
training, you can keep data anywhere and pass the folders explicitly with the
stage command-line arguments.

## Clinical Tumor Localization Note

In the intended clinical workflow, physicians provide the primary tumor location
and side, and automatic primary-tumor segmentation can then be applied around
that clinical localization. In this code release, to keep the testing pipeline
more automatic, `LNA-Seg-Preprocess` uses TotalSegmentator to segment
tumor/nodule candidates, and `LNA-Dx-Preprocess` keeps the largest candidate.
When a case table does not provide `tumor_side`, `LNA-Dx-Preprocess` infers
`left`, `right`, or `other` from the spatial relationship between that largest
tumor/nodule candidate and the lung-lobe labels.

## Environments

The tested local setup used two environments, so this release keeps two root
dependency files instead of separate requirements files inside each subfolder.
The pins are references from the working local environments; adjust the PyTorch
CUDA build for your own machine if needed.

Stage 1:

```bash
conda create -n lna-totalseg python=3.10
conda activate lna-totalseg
pip install -r requirements-totalseg.txt
```

Stages 2-4:

```bash
conda create -n lna-pytorch python=3.8
conda activate lna-pytorch
pip install -r requirements-pytorch.txt
```

## Quick Start

Place local CT files in `LNA-Inputs/case_data_nii/`. If you have physician
provided tumor side or N-stage labels for local testing, copy
`LNA-Inputs/case_info/case_table_template.csv` to
`LNA-Inputs/case_info/case_table_local.csv` and edit it locally.

Run the four stages from the repository root.

### Stage 1: Segmentation Preprocess

Use the TotalSegmentator environment:

```bash
python LNA-Seg-Preprocess/run_preprocess.py \
  --input_dir LNA-Inputs/case_data_nii \
  --output_dir LNA-Outputs/01_preprocess
```

Main outputs:

```text
LNA-Outputs/01_preprocess/
  nnUNet_input/
  prior_full/
  prior_crop/
  prior_crop_onehot_5structure/
  lung_bbox_dict.json
  manifest.json
```

### Stage 2: LNA-Seg Inference

Use the PyTorch environment:

```bash
python LNA-Seg/run_lna_seg.py \
  --input_dir LNA-Inputs/case_data_nii \
  --work_dir LNA-Outputs/01_preprocess \
  --output_dir LNA-Outputs/02_lna_seg \
  --model_dir LNA-Seg/nnUnet_trained_models
```

Main output:

```text
LNA-Outputs/02_lna_seg/
  wPost_delTotal/
```

### Stage 3: VOI Generation

```bash
python LNA-Dx-Preprocess/run_make_voi.py \
  --input_dir LNA-Inputs/case_data_nii \
  --work_dir LNA-Outputs \
  --output_dir LNA-Outputs/03_voi_for_lna_dx
```

To force physician-provided side information, pass a table explicitly:

```bash
python LNA-Dx-Preprocess/run_make_voi.py \
  --input_dir LNA-Inputs/case_data_nii \
  --work_dir LNA-Outputs \
  --output_dir LNA-Outputs/03_voi_for_lna_dx \
  --case_table LNA-Inputs/case_info/case_table_local.csv
```

Main outputs:

```text
LNA-Outputs/03_voi_for_lna_dx/
  img_VOI/
  seg_VOI/
```

### Stage 4: LNA-Dx Inference

```bash
python LNA-Dx/test.py \
  --image-voi-dir LNA-Outputs/03_voi_for_lna_dx/img_VOI \
  --prior-voi-dir LNA-Outputs/03_voi_for_lna_dx/seg_VOI \
  --weights LNA-Dx/checkpoints/R18/weights_dx.pth \
  --output-file LNA-Outputs/04_lna_dx/Predictions_Dx.xlsx
```

If you want the output table to include ground-truth N-stage values:

```bash
python LNA-Dx/test.py \
  --image-voi-dir LNA-Outputs/03_voi_for_lna_dx/img_VOI \
  --prior-voi-dir LNA-Outputs/03_voi_for_lna_dx/seg_VOI \
  --weights LNA-Dx/checkpoints/R18/weights_dx.pth \
  --gt-table path/to/case_table.csv \
  --gt-column n_stage \
  --output-file LNA-Outputs/04_lna_dx/Predictions_Dx.xlsx
```

Main outputs:

```text
LNA-Outputs/04_lna_dx/
  Predictions_Dx.xlsx
  gs-cam1_VOI/
```

## Training

### LNA-Seg Training

The segmentation training code is the vendored nnU-Net v1 code under
`LNA-Seg/nnunet`. Preprocess the segmentation training CT volumes with
`LNA-Seg-Preprocess`, pair the generated nnU-Net-style image channels with your
lymph-node labels in a standard nnU-Net dataset layout, then train with the
standard nnU-Net v1 training entry points in `LNA-Seg/nnunet/run/`. After
training, place the trained model output under `LNA-Seg/nnUnet_trained_models/`
for Stage 2 inference.

### LNA-Dx Training

After Stage 3 has generated diagnosis VOIs, train the diagnosis model with your
own label table:

```bash
python LNA-Dx/train.py \
  --image-voi-dir path/to/img_VOI \
  --prior-voi-dir path/to/seg_VOI \
  --label-table path/to/dx_labels.csv \
  --label-column label
```

The label table must include `case_id` and a diagnosis label column. Labels lower
than 1 are treated as non-metastatic, and labels greater than or equal to 1 are
treated as metastatic. You can also use the legacy fold-list interface with
`--image-list-dir`, `--prior-list-dir`, and `--train-folds`.

