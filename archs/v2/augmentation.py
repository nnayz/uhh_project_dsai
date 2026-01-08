"""
Audio-specific data augmentation for bioacoustic classification.

Implements SpecAugment and other audio augmentation techniques proven
to improve few-shot learning performance.

Based on:
- SpecAugment: Park et al., 2019
- Mixup: Zhang et al., 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Tuple


class TimeMasking(nn.Module):
    """
    Mask contiguous time steps in spectrogram.

    Args:
        max_mask_pct: Maximum percentage of time dimension to mask (0.0-1.0)
        num_masks: Number of time masks to apply
    """

    def __init__(self, max_mask_pct: float = 0.15, num_masks: int = 1):
        super().__init__()
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (batch, channels, freq, time)
        Returns:
            Augmented spectrogram
        """
        if not self.training:
            return spec

        batch, channels, freq, time = spec.shape
        max_mask_len = int(time * self.max_mask_pct)

        spec = spec.clone()
        for _ in range(self.num_masks):
            mask_len = random.randint(0, max_mask_len)
            if mask_len == 0:
                continue
            mask_start = random.randint(0, time - mask_len)
            spec[:, :, :, mask_start : mask_start + mask_len] = 0

        return spec


class FrequencyMasking(nn.Module):
    """
    Mask contiguous frequency bands in spectrogram.

    Args:
        max_mask_pct: Maximum percentage of frequency dimension to mask
        num_masks: Number of frequency masks to apply
    """

    def __init__(self, max_mask_pct: float = 0.15, num_masks: int = 1):
        super().__init__()
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (batch, channels, freq, time)
        Returns:
            Augmented spectrogram
        """
        if not self.training:
            return spec

        batch, channels, freq, time = spec.shape
        max_mask_len = int(freq * self.max_mask_pct)

        spec = spec.clone()
        for _ in range(self.num_masks):
            mask_len = random.randint(0, max_mask_len)
            if mask_len == 0:
                continue
            mask_start = random.randint(0, freq - mask_len)
            spec[:, :, mask_start : mask_start + mask_len, :] = 0

        return spec


class SpecAugment(nn.Module):
    """
    Complete SpecAugment implementation combining time and frequency masking.

    Args:
        time_mask_pct: Maximum percentage of time to mask
        freq_mask_pct: Maximum percentage of frequency to mask
        num_time_masks: Number of time masks
        num_freq_masks: Number of frequency masks
    """

    def __init__(
        self,
        time_mask_pct: float = 0.15,
        freq_mask_pct: float = 0.15,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        super().__init__()
        self.time_masking = TimeMasking(time_mask_pct, num_time_masks)
        self.freq_masking = FrequencyMasking(freq_mask_pct, num_freq_masks)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram."""
        spec = self.time_masking(spec)
        spec = self.freq_masking(spec)
        return spec


class MixUp(nn.Module):
    """
    Mixup data augmentation for few-shot learning.

    Mixes two spectrograms and their labels with a random weight.

    Args:
        alpha: Beta distribution parameter for mixup weight
    """

    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, spec: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply mixup augmentation.

        Args:
            spec: (batch, channels, freq, time)
            labels: (batch,) class labels

        Returns:
            Mixed spectrogram and labels
        """
        if not self.training or spec.size(0) < 2:
            return spec, labels

        # Sample mixup weight from beta distribution
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = spec.size(0)
        index = torch.randperm(batch_size, device=spec.device)

        # Mix spectrograms
        mixed_spec = lam * spec + (1 - lam) * spec[index]

        # Mix labels if provided (for loss computation)
        if labels is not None:
            # Return both labels and mixing weight for loss computation
            mixed_labels = (labels, labels[index], lam)
            return mixed_spec, mixed_labels

        return mixed_spec, None


class RandomGaussianNoise(nn.Module):
    """
    Add random Gaussian noise to simulate recording variations.

    Args:
        noise_std: Standard deviation of Gaussian noise
        prob: Probability of applying noise
    """

    def __init__(self, noise_std: float = 0.01, prob: float = 0.3):
        super().__init__()
        self.noise_std = noise_std
        self.prob = prob

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to spectrogram."""
        if not self.training or random.random() > self.prob:
            return spec

        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise


class BioacousticAugmentation(nn.Module):
    """
    Complete augmentation pipeline for bioacoustic few-shot learning.

    Combines SpecAugment, Mixup, and noise augmentation.

    Args:
        use_spec_augment: Enable SpecAugment
        use_mixup: Enable Mixup
        use_noise: Enable Gaussian noise
        time_mask_pct: SpecAugment time masking percentage
        freq_mask_pct: SpecAugment frequency masking percentage
        mixup_alpha: Mixup alpha parameter
    """

    def __init__(
        self,
        use_spec_augment: bool = True,
        use_mixup: bool = False,  # Mixup disabled by default (complex with episodes)
        use_noise: bool = True,
        time_mask_pct: float = 0.15,
        freq_mask_pct: float = 0.15,
        mixup_alpha: float = 0.4,
    ):
        super().__init__()

        self.use_spec_augment = use_spec_augment
        self.use_mixup = use_mixup
        self.use_noise = use_noise

        if use_spec_augment:
            self.spec_augment = SpecAugment(
                time_mask_pct=time_mask_pct,
                freq_mask_pct=freq_mask_pct,
            )

        if use_mixup:
            self.mixup = MixUp(alpha=mixup_alpha)

        if use_noise:
            self.noise = RandomGaussianNoise()

    def forward(
        self, spec: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation pipeline.

        Args:
            spec: (batch, channels, freq, time)
            labels: Optional labels for mixup

        Returns:
            Augmented spectrogram and labels
        """
        # SpecAugment (most important for audio)
        if self.use_spec_augment:
            spec = self.spec_augment(spec)

        # Gaussian noise
        if self.use_noise:
            spec = self.noise(spec)

        # Mixup (optional, can complicate few-shot learning)
        if self.use_mixup and labels is not None:
            spec, labels = self.mixup(spec, labels)

        return spec, labels
