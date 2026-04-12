#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frame-level temporal tactile encoder (v2).

Key differences from v1 (point-level):
- Each frame's 44 points are flattened and projected as a single token (frame_embed).
- Only temporal positional encoding is used (no spatial pos_embed).
- Dynamic time slicing: supports variable-length input sequences up to max_seq_len.
- Output shape: (B, T, embed_dim)  instead of v1's (B, T*N, embed_dim).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor, nn


class TemporalTactileEncoderV2(nn.Module):
    """Frame-level temporal transformer encoder for sparse point-cloud tactile input.

    Expected input shape: (B, T, num_points, point_dim).
    Returns token embeddings of shape (B, T, embed_dim).
    """

    def __init__(
        self,
        num_points: int = 44,
        point_dim: int = 4,
        embed_dim: int = 384,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.frame_embed = nn.Linear(num_points * point_dim, embed_dim)
        self.time_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, sparse_history: Tensor) -> Tensor:
        # sparse_history: [B, T, num_points, point_dim]
        B, T, N, D = sparse_history.shape
        x = sparse_history.reshape(B, T, N * D)  # [B, T, num_points * point_dim]
        x = self.frame_embed(x)  # [B, T, embed_dim]
        # Dynamic time slicing: always align the latest frame to the last position.
        x = x + self.time_embed[:, -T:, :]
        return self.transformer(x)  # [B, T, embed_dim]


class PretrainedSparsePCEncoderV2(nn.Module):
    """Wrapper to load and optionally freeze the v2 pretrained temporal tactile encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        max_seq_len: int,
        num_points: int = 44,
        point_dim: int = 4,
        embed_dim: int = 384,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = TemporalTactileEncoderV2(
            num_points=num_points,
            point_dim=point_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if freeze:
            self.freeze()

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained v2 encoder checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
            raw_state_dict = checkpoint["encoder_state_dict"]
        elif isinstance(checkpoint, dict):
            raw_state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format in {path}")

        clean_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in raw_state_dict.items())
        self.encoder.load_state_dict(clean_state_dict, strict=True)

    def freeze(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, sparse_history: Tensor) -> Tensor:
        if self.training and any(p.requires_grad for p in self.encoder.parameters()):
            return self.encoder(sparse_history)
        with torch.no_grad():
            return self.encoder(sparse_history)
