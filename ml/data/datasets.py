"""
PyTorch datasets for DisgraPhi training stages.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class IAMDataset(Dataset):
    """
    PyTorch dataset for IAM Handwriting Database.
    Used for bootstrap training stage.

    Args:
        data_dir: Root directory containing processed IAM data
        split: One of 'train', 'val', 'test'
        transform: Optional image transforms
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load line image paths and ground truth
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load image paths and corresponding ground truth text."""
        samples = []

        split_file = self.data_dir / 'splits' / f'{self.split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            for line in f:
                line_id = line.strip()
                img_path = self.data_dir / 'lines' / f'{line_id}.png'
                txt_path = self.data_dir / 'ground_truth' / f'{line_id}.txt'

                if img_path.exists() and txt_path.exists():
                    with open(txt_path, 'r') as tf:
                        text = tf.read().strip()
                    samples.append({
                        'image_path': str(img_path),
                        'text': text,
                        'line_id': line_id
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': sample['text'],
            'line_id': sample['line_id']
        }


class PersonalizationDataset(Dataset):
    """
    PyTorch dataset for user-specific personalization training.
    Uses processed line crops from photographed packets.

    Args:
        user_dir: Directory containing user's processed packet data
        transform: Optional image transforms
    """

    def __init__(
        self,
        user_dir: str,
        transform=None
    ):
        self.user_dir = Path(user_dir)
        self.transform = transform

        # Load line crops and ground truth
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load cropped line images and corresponding ground truth."""
        samples = []

        lines_dir = self.user_dir / 'lines'
        ground_truth_file = self.user_dir / 'ground_truth.json'

        if not lines_dir.exists() or not ground_truth_file.exists():
            raise FileNotFoundError(
                f"User data not found in {self.user_dir}. "
                "Ensure packet has been processed."
            )

        import json
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        for line_id, text in ground_truth.items():
            img_path = lines_dir / f'{line_id}.png'
            if img_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'text': text,
                    'line_id': line_id
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': sample['text'],
            'line_id': sample['line_id']
        }
