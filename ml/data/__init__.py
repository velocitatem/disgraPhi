"""
Data pipeline for DisgraPhi handwriting recognition system.

This module provides:
- IAM Handwriting Database downloader with caching
- Personalization packet processing (QR detection, de-skewing, line cropping)
- PyTorch datasets for bootstrap and personalization training stages
"""

from .etl import IAMDownloader, PacketProcessor
from .datasets import IAMDataset, PersonalizationDataset

__all__ = [
    'IAMDownloader',
    'PacketProcessor',
    'IAMDataset',
    'PersonalizationDataset',
]
