from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lna_dx.utils import AverageMeter, unwrap_model


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.2, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tversky = (tp + eps) / (tp + (1 - alpha) * fp + alpha * fn + eps)
    return 1.0 - tversky


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = torch.sum(pred * target)
    return 1.0 - (2.0 * intersection + eps) / (torch.sum(pred) + torch.sum(target) + eps)


class AttributionDisentanglementLoss(nn.Module):
    """DAR-style loss used to constrain class-specific GS-CAM maps."""

    def forward(self, cam0: torch.Tensor, cam1: torch.Tensor, prior: torch.Tensor):
        prior = torch.round(prior * 26)
        losses = []
        label0_losses = []
        label1_losses = []
        dice_losses = []
        mutual_losses = []

        for batch_index in range(cam0.shape[0]):
            cam0_b = cam0[batch_index]
            cam1_b = cam1[batch_index]
            cam01_b = torch.clamp(cam0_b + cam1_b, min=0.0, max=1.0)
            prior_b = prior[batch_index]

            tumor_mask = (prior_b == 26).float()
            ln_non_mediastinal_mask = (
                ((prior_b >= 10) & (prior_b <= 14)) | ((prior_b >= 21) & (prior_b <= 25))
            ).float()
            ln_mediastinal_mask = (
                ((prior_b >= 1) & (prior_b <= 9)) | ((prior_b >= 15) & (prior_b <= 20))
            ).float()
            tumor_ln_mask = tumor_mask + ln_non_mediastinal_mask + ln_mediastinal_mask
            ln_mask = ln_non_mediastinal_mask + ln_mediastinal_mask

            loss_label0 = tversky_loss(cam0_b, ln_mask, alpha=0.1)
            loss_label1 = tversky_loss(cam1_b, tumor_ln_mask, alpha=0.1)
            loss_dice = dice_loss(cam01_b, tumor_ln_mask)

            overlap_region = ln_mask > 0
            if torch.any(overlap_region):
                overlap = cam0_b * cam1_b
                loss_mutual = (overlap * (cam0_b + cam1_b))[overlap_region].mean()
            else:
                loss_mutual = cam0_b.new_tensor(0.0)

            loss = (loss_label0 + loss_label1 + loss_dice + loss_mutual) / 4.0
            losses.append(loss)
            label0_losses.append(loss_label0)
            label1_losses.append(loss_label1)
            dice_losses.append(loss_dice)
            mutual_losses.append(loss_mutual)

        return (
            torch.stack(losses).mean(),
            torch.stack(label0_losses).mean(),
            torch.stack(label1_losses).mean(),
            torch.stack(dice_losses).mean(),
            torch.stack(mutual_losses).mean(),
        )


def classifier_weight(model: torch.nn.Module) -> torch.Tensor:
    return unwrap_model(model).linear_cls.weight


def register_feature_hook(model: torch.nn.Module, feature_blobs: list[torch.Tensor]):
    def hook_feature(_module, _input, output):
        feature_blobs.append(output)

    return unwrap_model(model).relu_out.register_forward_hook(hook_feature)


def gs_cam(feature_conv: torch.Tensor, weight_softmax: torch.Tensor, class_index: int) -> torch.Tensor:
    batch_size, channels, depth, height, width = feature_conv.shape
    selected_weight = weight_softmax[class_index].view(1, 1, channels).expand(batch_size, -1, -1)
    flattened = feature_conv.reshape(batch_size, channels, depth * height * width)
    cam = torch.bmm(selected_weight, flattened).reshape(batch_size, depth, height, width)
    return F.relu(torch.sigmoid(cam) - 0.5) * 2.0


def resize_prior_to_cam(prior: torch.Tensor, cam_shape: tuple[int, int, int]) -> torch.Tensor:
    prior = prior.unsqueeze(1)
    prior = F.interpolate(prior, size=cam_shape, mode="nearest")
    return prior.squeeze(1)


def run_train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    loss_weight: float,
    steps_per_epoch: int | None = None,
) -> float:
    model.train()
    ce_loss = nn.CrossEntropyLoss().to(device)
    ad_loss = AttributionDisentanglementLoss().to(device)
    losses = AverageMeter()

    feature_blobs: list[torch.Tensor] = []
    hook = register_feature_hook(model, feature_blobs)
    start_time = time.time()

    try:
        for step, (images, labels) in enumerate(loader, start=1):
            if steps_per_epoch is not None and step > steps_per_epoch:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feature_blobs.clear()

            logits = model(images)
            if not feature_blobs:
                raise RuntimeError("Feature hook did not capture relu_out activations.")

            weights = classifier_weight(model)
            cam0 = gs_cam(feature_blobs[0], weights, 0)
            cam1 = gs_cam(feature_blobs[0], weights, 1)
            prior = resize_prior_to_cam(images[:, 2], cam0.shape[-3:])

            cls_loss = ce_loss(logits, labels)
            cam_loss, *_ = ad_loss(cam0, cam1, prior)
            total_loss = loss_weight * cls_loss + cam_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.update(total_loss.item(), images.size(0))
    finally:
        hook.remove()

    scheduler.step()
    print(f"Train time: {time.time() - start_time:.4f}s")
    return losses.avg


def run_validation_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    loss_weight: float,
) -> float:
    model.eval()
    ce_loss = nn.CrossEntropyLoss().to(device)
    ad_loss = AttributionDisentanglementLoss().to(device)
    losses = AverageMeter()

    feature_blobs: list[torch.Tensor] = []
    hook = register_feature_hook(model, feature_blobs)

    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feature_blobs.clear()

                logits = model(images)
                if not feature_blobs:
                    raise RuntimeError("Feature hook did not capture relu_out activations.")

                weights = classifier_weight(model)
                cam0 = gs_cam(feature_blobs[0], weights, 0)
                cam1 = gs_cam(feature_blobs[0], weights, 1)
                prior = resize_prior_to_cam(images[:, 2], cam0.shape[-3:])

                cls_loss = ce_loss(logits, labels)
                cam_loss, *_ = ad_loss(cam0, cam1, prior)
                total_loss = loss_weight * cls_loss + cam_loss
                losses.update(total_loss.item(), images.size(0))
    finally:
        hook.remove()

    return losses.avg


def save_checkpoint(model: torch.nn.Module, output_dir: str | Path, filename: str = "weights_cls.pth") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    torch.save({"state_dict": unwrap_model(model).state_dict()}, output_path)
    return output_path
