"""Curriculum sampler: progressively include harder examples during training.

Difficulty is measured as the divergence score from a pre-trained baseline
(or a heuristic: short utterances are easy, long ambiguous ones are hard).

Strategy:
  - Epoch 0..warmup_epochs: only easy examples (difficulty ≤ threshold)
  - Epoch warmup..max_epochs: threshold linearly increases from 0 → 1
  - After max_epochs: all examples included
"""

from __future__ import annotations

from typing import List

import numpy as np
from torch.utils.data import Sampler


class CurriculumSampler(Sampler):
    """Difficulty-based curriculum sampler.

    Args:
        difficulties: Per-example difficulty scores in [0, 1].
                      Compute these from a pre-trained baseline model's
                      predicted divergence on the training set.
        epoch: Current training epoch (updated via set_epoch).
        max_epochs: Total training epochs.
        warmup_epochs: Epochs using only easy examples.
        shuffle: Whether to shuffle within the selected subset.
    """

    def __init__(
        self,
        difficulties: List[float],
        epoch: int = 0,
        max_epochs: int = 30,
        warmup_epochs: int = 5,
        shuffle: bool = True,
    ):
        super().__init__()
        self.difficulties = np.array(difficulties)
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.shuffle = shuffle

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def _threshold(self) -> float:
        if self.epoch < self.warmup_epochs:
            return 0.33  # easy 1/3 only
        progress = (self.epoch - self.warmup_epochs) / max(
            self.max_epochs - self.warmup_epochs, 1
        )
        return min(0.33 + 0.67 * progress, 1.0)  # ramp from 0.33 → 1.0

    def __iter__(self):
        threshold = self._threshold
        indices = np.where(self.difficulties <= threshold)[0]
        if len(indices) == 0:
            indices = np.arange(len(self.difficulties))
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        threshold = self._threshold
        return int((self.difficulties <= threshold).sum())
