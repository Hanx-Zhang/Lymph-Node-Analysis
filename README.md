# Lymph Node Analysis

This repository contains code for the Lymph Node Analysis (LNA) framework
described in the manuscript "Anatomically Grounded, Scalable, and Interpretable
CT Analysis of Thoracic Lymph Node Metastasis in Lung Cancer".

LNA is organized as two linked components:

1. `LNA-Seg`: fine-grained thoracic lymph node semantic segmentation.
2. `LNA-Dx`: interpretable patient-level lymph node metastasis diagnosis from
   CT and tumor-lymph-node anatomical priors.

The diagnosis component uses two CT intensity channels and one anatomical prior
channel. The first CT channel is a lung-window normalization, the second is a
mediastinal-window normalization, and the third channel is a tumor/LN prior map
derived from segmentation.

## Repository layout

```text
Lymph-Node-Analysis/
  README.md
  requirements.txt
  PAPER_CODE_NOTES.md
  LNA-Dx/
    main.py
    test.py
    lna_dx/
    requirements.txt
  LNA-Seg/
    README.md
    predict_local.py
    nnunet/
    test_lymph/
```

## LNA-Seg

`LNA-Seg` is built on nnU-Net v1 and maps thoracic CT volumes to station-level
lymph node semantic maps. The segmentation README includes the public
segmentation data description and nnU-Net installation notes.

The segmentation training data described there includes:

- TCIA mediastinal lymph node CT cases with refined annotations.
- St. Olavs Hospital contrast-enhanced CT cases with expert-refined lymph node
  annotations.

See `LNA-Seg/README.md` for details. The current segmentation folder is kept as
provided and was not modified during this cleanup pass.

## LNA-Dx

- direct training entry point: `LNA-Dx/main.py`
- direct inference entry point: `LNA-Dx/test.py`
- compact 3D ResNet with group normalization
- attribution-disentanglement regularization for GS-CAM-style maps
- optional CAM export during inference


## Installation

Install the diagnosis dependencies:

```bash
pip install -r requirements.txt
```

For segmentation, install nnU-Net v1 as described in `LNA-Seg/README.md`.

## Data organization

This repository does not include medical images, labels, pretrained weights, or
trained checkpoints. Large data and weight files should be stored outside Git
and passed to the scripts with command-line arguments.

The diagnosis scripts expect fold text files. Each row should contain:

```text
/path/to/image_volume.nii.gz label
```

The image-list and prior-list files must have the same row order. Labels are
converted to a binary metastasis target: labels lower than 1 are treated as
non-metastatic, and labels greater than or equal to 1 are treated as metastatic.

## Quick diagnosis commands

```bash
cd LNA-Dx
python main.py --no-pretrain
python test.py --weights checkpoints/R18/weights_cls.pth
```

Use `--image-list-dir`, `--prior-list-dir`, `--train-folds`, and `--test-folds`
to point the scripts to your own fold files.

## Notes for reviewers

`PAPER_CODE_NOTES.md` summarizes the current paper-code correspondence and the
known differences between the manuscript scope and this code release.
