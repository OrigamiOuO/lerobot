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

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor, nn


class TemporalTactileEncoder(nn.Module):
    """Pretrained temporal transformer encoder for sparse point-cloud tactile input.

    Expected input shape: (B, seq_len, num_points, in_dim).
    Returns token embeddings of shape (B, seq_len * num_points, embed_dim).
    """

    def __init__(self, in_dim: int = 4, embed_dim: int = 384, seq_len: int = 5, num_points: int = 44):
        super().__init__()
        self.point_embed = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_points, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, sparse_history: Tensor) -> Tensor:
        x = self.point_embed(sparse_history)
        x = x + self.pos_embed + self.time_embed
        x = x.flatten(1, 2)
        return self.transformer(x)


class PretrainedSparsePCEncoder(nn.Module):
    """Wrapper to load and optionally freeze the pretrained temporal tactile encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        seq_len: int,
        num_points: int,
        in_dim: int = 4,
        embed_dim: int = 384,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = TemporalTactileEncoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_points=num_points,
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if freeze:
            self.freeze()

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location="cpu")
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
