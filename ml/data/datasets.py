"""PyTorch datasets for DisgraPhi training stages."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ml.data.augmentation import HandwritingAugmentation
from ml.data.etl import IAMDownloader


class IAMDataset(Dataset):
    """
    PyTorch dataset for IAM Handwriting Database.
    Used for bootstrap training stage.

    Args:
        data_dir: Root directory containing processed IAM data
        split: One of 'train', 'val', 'test'
        transform: Optional image transforms
        sample_ratio: Fraction of dataset to use (0.0-1.0). Default 1.0 uses all data.
                     Useful for faster iteration during development.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        sample_ratio: float = 1.0,
        seed: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.sample_ratio = sample_ratio
        self.seed = seed if seed is not None else 2025

        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

        # Resolve directory and download IAM data if necessary
        self._ensure_data_ready()

        # Load line image paths and ground truth
        self.samples = self._load_samples()

    def _ensure_data_ready(self) -> None:
        """Ensure IAM processed data exists, downloading if missing."""

        processed_candidate = self.data_dir

        # Allow users to pass either the processed directory or the root
        if processed_candidate.is_dir() and (processed_candidate / 'splits').exists():
            return

        if not processed_candidate.exists() and processed_candidate.name == 'processed':
            processed_candidate = processed_candidate.parent

        if (processed_candidate / 'processed' / 'splits').exists():
            self.data_dir = processed_candidate / 'processed'
            return

        # Trigger automatic download + processing
        root_dir = processed_candidate if processed_candidate.name != 'processed' else processed_candidate.parent
        downloader = IAMDownloader(data_dir=str(root_dir))

        try:
            print("IAMDataset: processed data not found – downloading via IAMDownloader...")
            downloader.download(force=False)
            downloader.process()
        except Exception as exc:  # pragma: no cover - requires external service
            raise RuntimeError(
                "Failed to automatically prepare IAM dataset. "
                "Ensure HF_TOKEN is set and IAM credentials are valid."
            ) from exc

        self.data_dir = downloader.processed_dir

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

        # Apply sampling if requested
        if self.sample_ratio < 1.0:
            original_count = len(samples)
            # Use numpy for reproducible random sampling
            rng = np.random.default_rng(self.seed)
            n_samples = int(len(samples) * self.sample_ratio)
            indices = rng.choice(len(samples), size=n_samples, replace=False)
            samples = [samples[i] for i in sorted(indices)]
            print(f"Downsampled {self.split} set: {original_count} -> {len(samples)} samples ({self.sample_ratio*100:.0f}%)")

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
            subset = 'all'
            split_ratios = (0.8, 0.1, 0.1)

            if split is not None or split_type is not None:
                subset = split_type if split_type is not None else 'all'
                train_ratio = split if split is not None else 0.8
                val_ratio = max(0.0, 1.0 - train_ratio)
                split_ratios = (train_ratio, val_ratio, 0.0)

            self._dataset = ManifestDataset(
                data_dir=user_dir,
                transform=transform,
                subset=subset,
                split_ratios=split_ratios,
                seed=2025
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
        augment: Enable handwriting-specific augmentation (for few-shot training)
        augment_strength: Augmentation intensity 0.0-1.0 (default: 0.7)
        augment_prob: Probability of applying each augmentation (default: 0.5)
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        subset: str = 'all',
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: Optional[int] = None,
        augment: bool = False,
        augment_strength: float = 0.7,
        augment_prob: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.subset = subset
        self.split_ratios = tuple(split_ratios)
        self.seed = seed if seed is not None else 2025
        self.augment = augment

        # Setup augmentation pipeline
        if augment and subset == 'train':
            self.augmenter = HandwritingAugmentation(
                strength=augment_strength,
                prob=augment_prob
            )
        else:
            self.augmenter = None

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

        # Apply train/val/test split if requested
        if self.subset != 'all':
            if len(self.split_ratios) != 3:
                raise ValueError("split_ratios must contain three values (train, val, test)")

            if not np.isclose(sum(self.split_ratios), 1.0, atol=1e-6):
                raise ValueError("split_ratios must sum to 1.0")

            n_samples = len(samples)
            if n_samples < 3:
                raise ValueError("At least 3 samples are required to compute train/val/test splits")

            indices = np.arange(n_samples)
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

            train_end = int(self.split_ratios[0] * n_samples)
            val_end = train_end + int(self.split_ratios[1] * n_samples)

            # Guarantee at least one sample per split when possible
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)
            val_end = min(val_end, n_samples - 1)

            if self.subset == 'train':
                chosen = indices[:train_end]
            elif self.subset == 'val':
                chosen = indices[train_end:val_end]
            elif self.subset == 'test':
                chosen = indices[val_end:]
            else:
                raise ValueError("subset must be one of {'train','val','test','all'}")

            samples = [samples[i] for i in sorted(chosen)]

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        subset_label = self.subset if self.subset != 'all' else 'full'
        print(f"Loaded {len(samples)} samples from {self.data_dir} [{subset_label}]")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Apply augmentation first (if enabled)
        if self.augmenter is not None:
            image = self.augmenter(image)

        # Apply user-provided transforms (if any)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': sample['text'],
            'line_id': f"sample_{sample['sample_id']:04d}"
        }
