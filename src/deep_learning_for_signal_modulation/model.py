"""Reference VT-CNN2 architecture (O'Shea et al., 2016; Project Specification Table 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VTCNN2Config:
    """Hyperparameters defining the reference VT-CNN2 architecture."""

    num_classes: int = 11
    signal_length: int = 128
    conv1_filters: int = 256
    conv2_filters: int = 80
    fc_units: int = 256
    dropout: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)


class VTCNN2(nn.Module):
    """PyTorch implementation of Table 1 Reference VT-CNN2 for RML2016.10a.

    Input shape: (batch, 1, 2, 128) — equivalent to Keras channels-last (batch, 2, 128, 1).
    No pooling layers are used so fine-grained temporal phase transitions are preserved.
    """

    def __init__(self, config: VTCNN2Config | None = None) -> None:
        super().__init__()
        self.config = config or VTCNN2Config()
        c = self.config

        # Pad width (time axis) by 2 on left and 2 on right: (2, 128) -> (2, 132)
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=c.conv1_filters,
            kernel_size=(1, 3),
            padding=0,
        )
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=c.dropout)

        # Pad width (time axis) by 2 on left and 2 on right: (2, 130) -> (2, 134)
        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(
            in_channels=c.conv1_filters,
            out_channels=c.conv2_filters,
            kernel_size=(2, 3),
            padding=0,
        )
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=c.dropout)

        self.flatten = nn.Flatten()
        # (2, 128) -> pad(0,2) -> (2, 132) -> conv(1,3) -> (2, 130) -> pad(0,2) -> (2, 134) -> conv(2,3) -> (1, 132)
        flattened_dim = c.conv2_filters * (c.signal_length + 4)
        self.fc1 = nn.Linear(flattened_dim, c.fc_units)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=c.dropout)
        self.fc2 = nn.Linear(c.fc_units, c.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Table 1 initializers: Glorot uniform for Conv2D/output, He normal for Dense(256)."""
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim penultimate latent representation (prior to final Dropout/Linear)."""
        x = self.pad1(x)
        x = self.drop1(self.act1(self.conv1(x)))
        x = self.pad2(x)
        x = self.drop2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits of shape (batch, num_classes)."""
        features = self.forward_features(x)
        return self.fc2(self.drop3(features))

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def architecture_summary(self) -> list[dict[str, object]]:
        """Trace layer output shapes, parameter counts, and multiply-accumulate ops (MACs)."""
        c = self.config
        rows: list[dict[str, object]] = []
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 1, 2, c.signal_length, device=device)

        def _conv_macs(layer: nn.Conv2d, out_tensor: torch.Tensor) -> int:
            _, out_c, out_h, out_w = out_tensor.shape
            k_h, k_w = layer.kernel_size
            return int(out_c * out_h * out_w * layer.in_channels * k_h * k_w)

        def _linear_macs(layer: nn.Linear) -> int:
            return int(layer.in_features * layer.out_features)

        def _params(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters()))

        with torch.no_grad():
            rows.append(
                {
                    "layer": "Input",
                    "details": "Normalized I/Q frame",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [2, c.signal_length, 1],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.pad1(x)
            rows.append(
                {
                    "layer": "ZeroPadding2D_1",
                    "details": "Padding (0, 2)",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.act1(self.conv1(x))
            rows.append(
                {
                    "layer": "Conv2D_1",
                    "details": f"{c.conv1_filters} filters, kernel (1x3), valid, ReLU, Glorot uniform",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": _params(self.conv1),
                    "macs": _conv_macs(self.conv1, x),
                }
            )

            x = self.drop1(x)
            rows.append(
                {
                    "layer": "Dropout_1",
                    "details": f"Rate: {c.dropout}",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.pad2(x)
            rows.append(
                {
                    "layer": "ZeroPadding2D_2",
                    "details": "Padding (0, 2)",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.act2(self.conv2(x))
            rows.append(
                {
                    "layer": "Conv2D_2",
                    "details": f"{c.conv2_filters} filters, kernel (2x3), valid, ReLU, Glorot uniform",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": _params(self.conv2),
                    "macs": _conv_macs(self.conv2, x),
                }
            )

            x = self.drop2(x)
            rows.append(
                {
                    "layer": "Dropout_2",
                    "details": f"Rate: {c.dropout}",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[2], x.shape[3], x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.flatten(x)
            rows.append(
                {
                    "layer": "Flatten",
                    "details": "-",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.act3(self.fc1(x))
            rows.append(
                {
                    "layer": "Dense_1 (FC)",
                    "details": f"{c.fc_units} units, ReLU, He normal",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[1]],
                    "params": _params(self.fc1),
                    "macs": _linear_macs(self.fc1),
                }
            )

            x = self.drop3(x)
            rows.append(
                {
                    "layer": "Dropout_3",
                    "details": f"Rate: {c.dropout}",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[1]],
                    "params": 0,
                    "macs": 0,
                }
            )

            x = self.fc2(x)
            rows.append(
                {
                    "layer": "Dense_2 (Output)",
                    "details": f"{c.num_classes} units, Logits (Softmax at eval), Glorot uniform",
                    "output_shape_nchw": list(x.shape),
                    "output_shape_keras_equiv": [x.shape[1]],
                    "params": _params(self.fc2),
                    "macs": _linear_macs(self.fc2),
                }
            )

        if was_training:
            self.train()
        return rows
