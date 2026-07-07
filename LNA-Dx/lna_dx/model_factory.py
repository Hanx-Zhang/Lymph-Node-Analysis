from __future__ import annotations

from argparse import Namespace

from lna_dx.models import resnet_groupnorm


def generate_model(args: Namespace):
    if args.model != "resnet_groupnorm":
        raise ValueError("This release contains the ResNet group-normalization backbone only.")

    builders = {
        10: resnet_groupnorm.resnet10,
        18: resnet_groupnorm.resnet18,
        34: resnet_groupnorm.resnet34,
        50: resnet_groupnorm.resnet50,
        101: resnet_groupnorm.resnet101,
        152: resnet_groupnorm.resnet152,
        200: resnet_groupnorm.resnet200,
    }
    if args.model_depth not in builders:
        raise ValueError(f"Unsupported ResNet depth: {args.model_depth}")

    return builders[args.model_depth](
        shortcut_type=args.resnet_shortcut,
        num_classes=args.num_classes,
    )
