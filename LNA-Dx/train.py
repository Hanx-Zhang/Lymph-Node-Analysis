from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from lna_dx_core.data import create_train_val_loaders
from lna_dx_core.engine import run_train_epoch, run_validation_epoch, save_checkpoint
from lna_dx_core.model_factory import generate_model
from lna_dx_core.utils import checkpoint_state_dict, dx_root, strip_module_prefix, unwrap_model


REPO_ROOT = dx_root().parent
DEFAULT_DX_VOI_ROOT = REPO_ROOT / "LNA-Outputs" / "03_voi_for_lna_dx"
DEFAULT_IMAGE_VOI_DIR = DEFAULT_DX_VOI_ROOT / "img_VOI"
DEFAULT_PRIOR_VOI_DIR = DEFAULT_DX_VOI_ROOT / "seg_VOI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LNA-Dx metastasis diagnosis model.")

    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--savedir", default="R18", help="Checkpoint subdirectory name.")
    parser.add_argument("--model", default="resnet_groupnorm", choices=["resnet_groupnorm"])
    parser.add_argument("--model-depth", type=int, default=18, choices=[10, 18, 34, 50, 101, 152, 200])
    parser.add_argument("--num-classes", type=int, default=2, help="Diagnosis classes: 0 non-metastatic, 1 metastatic.")
    parser.add_argument("--resnet-shortcut", default="B", choices=["A", "B"])

    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=dx_root() / "pretrain" / "weights_pre.pth",
        help="Path to malignancy-pretrained weights. Use --no-pretrain to disable.",
    )
    parser.add_argument("--no-pretrain", action="store_true", help="Train without loading pretrained weights.")

    parser.add_argument("--image-voi-dir", type=Path, default=DEFAULT_IMAGE_VOI_DIR)
    parser.add_argument("--prior-voi-dir", type=Path, default=DEFAULT_PRIOR_VOI_DIR)
    parser.add_argument("--label-table", type=Path, default=None, help="CSV/XLSX with case_id and diagnosis labels.")
    parser.add_argument("--label-column", default=None, help="Diagnosis label column name in --label-table.")
    parser.add_argument("--case-id-column", default="case_id", help="Case id column name in --label-table.")
    parser.add_argument("--image-list-dir", type=Path, default=None, help="Legacy fold-list image directory.")
    parser.add_argument("--prior-list-dir", type=Path, default=None, help="Legacy fold-list prior directory.")
    parser.add_argument("--train-folds", default="1", help="Legacy fold ids joined by '-', for example '1-2-3'.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--balanced-sampling",
        dest="balanced_sampling",
        action="store_true",
        default=True,
        help="Use inverse-frequency balanced sampling for the fine-tuning training loader.",
    )
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", default="ExponentialLR", choices=["ExponentialLR", "StepLR"])
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2000,
        help="Training batches per epoch.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use '' for the runtime default.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=dx_root() / "checkpoints",
        help="Directory where training checkpoints are written.",
    )
    return parser.parse_args()


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def run_training(args: argparse.Namespace, run_id: int) -> None:
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = use_cuda

    model = generate_model(args)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    if not args.no_pretrain:
        if not args.pretrained_weights.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found: {args.pretrained_weights}. "
                "Pass --no-pretrain or provide --pretrained-weights."
        )
        checkpoint_pretrain = torch.load(args.pretrained_weights, map_location="cpu")
        incoming = strip_module_prefix(checkpoint_state_dict(checkpoint_pretrain))
        current = unwrap_model(model).state_dict()
        matched = {
            key: value
            for key, value in incoming.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        current.update(matched)
        unwrap_model(model).load_state_dict(current)
        loaded, total = len(matched), len(incoming)
        print(f"Loaded {loaded}/{total} compatible pretrained tensors from {args.pretrained_weights}")

    train_loader, val_loader = create_train_val_loaders(
        image_list_dir=args.image_list_dir,
        prior_list_dir=args.prior_list_dir,
        folds=args.train_folds,
        image_voi_dir=args.image_voi_dir,
        prior_voi_dir=args.prior_voi_dir,
        label_table=args.label_table,
        label_column=args.label_column,
        case_id_column=args.case_id_column,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        augment=args.augment,
        balanced_sampling=args.balanced_sampling,
        seed=args.seed,
    )

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    output_dir = args.checkpoint_root / args.savedir
    if args.runs > 1:
        output_dir = output_dir / f"run_{run_id}"

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}.")
        train_loss = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_weight=args.loss_weight,
            steps_per_epoch=args.steps_per_epoch,
        )
        val_loss = run_validation_epoch(model, val_loader, device, loss_weight=args.loss_weight)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            saved_path = save_checkpoint(model, output_dir)
            print(f"Saved checkpoint: {saved_path}")

        if epoch + 1 > best_epoch + 10:
            print("Early stopping after 10 epochs without validation improvement.")
            break

    print(f"Training finished in {time.time() - start_time:.4f}s")


def main() -> None:
    args = parse_args()
    for run_id in range(args.runs):
        run_training(args, run_id)


if __name__ == "__main__":
    main()
