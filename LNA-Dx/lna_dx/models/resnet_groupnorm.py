from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x: torch.Tensor, planes: int, stride: int, no_cuda: bool = False) -> torch.Tensor:
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device if not no_cuda else torch.device("cpu"),
    )
    if zero_pads.device != out.device:
        zero_pads = zero_pads.to(out.device)
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(8, planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(8, planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual


class ResNet(nn.Module):
    """Compact 3D ResNet used by LNA-Dx."""

    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: list[int],
        *,
        shortcut_type: str = "B",
        no_cuda: bool = False,
        in_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda

        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.gn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[1], shortcut_type, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer3 = self._make_layer(block, 256, layers[3], shortcut_type, stride=1, dilation=4)
        self.relu_out = nn.ReLU(inplace=False)
        self.linear_cls = nn.Linear(256 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(8, planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                module.bias.data.zero_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return self.relu_out(x)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = F.adaptive_avg_pool3d(features, (1, 1, 1))
        pooled = pooled.view(pooled.size(0), -1)
        return self.linear_cls(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, 2] = (x[:, 2] > 0).float()
        return self.forward_once(x[:, 0:3])


def resnet10(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet200(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
