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
    DEPRECATED: Use ManifestDataset instead.

    PyTorch dataset for user-specific personalization training.
    This is now a wrapper around ManifestDataset for backward compatibility.

    The old format used:
    - lines/ directory with line_XXX.png files
    - ground_truth.json with {"line_id": "text"} mapping

    The new ManifestDataset format uses:
    - manifest.json with full metadata
    - sample_XXXX.png files

    Args:
        user_dir: Directory containing user's processed packet data
        transform: Optional image transforms
        split: Optional train/val split ratio (e.g., 0.8)
        split_type: 'train' or 'val' when using split
    """

    def __init__(
        self,
        user_dir: str,
        transform=None,
        split: Optional[float] = None,
        split_type: Optional[str] = None
    ):
        import warnings
        warnings.warn(
            "PersonalizationDataset is deprecated. Use ManifestDataset instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self.user_dir = Path(user_dir)

        # Check if using new manifest format
        manifest_path = self.user_dir / 'manifest.json'
        if manifest_path.exists():
            # Use ManifestDataset
            self._dataset = ManifestDataset(
                data_dir=user_dir,
                transform=transform,
                split=split,
                split_type=split_type
            )
        else:
            # Fallback to old format (for backward compatibility)
            self._dataset = None
            self.transform = transform
            self.samples = self._load_samples_old_format()

    def _load_samples_old_format(self) -> List[Dict[str, str]]:
        """Load samples from old format (lines/ + ground_truth.json)."""
        samples = []

        lines_dir = self.user_dir / 'lines'
        ground_truth_file = self.user_dir / 'ground_truth.json'

        if not lines_dir.exists() or not ground_truth_file.exists():
            raise FileNotFoundError(
                f"User data not found in {self.user_dir}. "
                "Ensure packet has been processed or use ManifestDataset for new format."
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

        print(f"Loaded {len(samples)} samples from old format: {self.user_dir}")
        return samples

    def __len__(self) -> int:
        if self._dataset is not None:
            return len(self._dataset)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        if self._dataset is not None:
            return self._dataset[idx]

        # Old format
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': sample['text'],
            'line_id': sample['line_id']
        }


class ManifestDataset(Dataset):
    """
    PyTorch dataset for handwriting data with manifest.json format.

    This is a generic dataset loader for any handwriting data organized as:
    - manifest.json: Contains metadata and ground truth
    - sample_XXXX.png: Image files

    Manifest format:
    {
        "created": "2025-10-24T18:32:12.480Z",
        "totalSamples": 11,
        "samples": [
            {
                "imageFile": "sample_0001.png",
                "groundTruth": "Text content here...",
                "index": 0
            },
            ...
        ]
    }

    Args:
        data_dir: Directory containing manifest.json and sample images
        transform: Optional image transforms
        split: Optional train/val split ratio (e.g., 0.8 = 80% train, 20% val)
        split_type: 'train' or 'val' when using split
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: Optional[float] = None,
        split_type: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.split_type = split_type

        # Load samples from manifest
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load samples from manifest.json."""
        manifest_path = self.data_dir / 'manifest.json'

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        samples = []
        for sample_data in manifest.get('samples', []):
            img_path = self.data_dir / sample_data['imageFile']

            if img_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'text': sample_data['groundTruth'],
                    'sample_id': sample_data.get('index', len(samples))
                })

        # Apply train/val split if requested
        if self.split is not None and self.split_type is not None:
            n_samples = len(samples)
            n_train = int(n_samples * self.split)

            if self.split_type == 'train':
                samples = samples[:n_train]
            elif self.split_type == 'val':
                samples = samples[n_train:]
            else:
                raise ValueError(f"split_type must be 'train' or 'val', got {self.split_type}")

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        print(f"Loaded {len(samples)} samples from {self.data_dir}")
        if self.split_type:
            print(f"  Split: {self.split_type} ({len(samples)} samples)")

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
            'line_id': f"sample_{sample['sample_id']:04d}"
        }
