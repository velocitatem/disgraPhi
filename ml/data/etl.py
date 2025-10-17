"""
ETL pipeline for DisgraPhi data processing.

Components:
- IAMDownloader: Download and cache IAM Handwriting Database
- PacketProcessor: Process user packets (QR detection, de-skew, line crop)
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
import cv2
import numpy as np
from pyzbar import pyzbar
import xml.etree.ElementTree as ET


class IAMDownloader:
    """
    Downloads and processes IAM Handwriting Database from Hugging Face.

    Downloads from private HF dataset: velocitatem/handwritten-baseline

    Args:
        data_dir: Root directory to store downloaded data
        hf_token: Hugging Face API token for private dataset access
        repo_id: HF dataset repository ID (default: velocitatem/handwritten-baseline)
    """

    # HF Dataset configuration
    DEFAULT_REPO_ID = "velocitatem/handwritten-baseline"

    def __init__(
        self,
        data_dir: str = "ml/data/raw/iam",
        hf_token: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.repo_id = repo_id or self.DEFAULT_REPO_ID

        # Create directory structure
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download(self, force: bool = False) -> None:
        """
        Download IAM database files from Hugging Face with caching.

        Args:
            force: If True, re-download even if files exist
        """
        if not self.hf_token:
            raise ValueError(
                "HF token not provided. Set HF_TOKEN environment variable "
                "or pass it to the constructor."
            )

        # Authenticate with Hugging Face
        print("Authenticating with Hugging Face...")
        login(token=self.hf_token)
        print("✓ Authentication successful")

        files_to_download = ['lines.tgz', 'ascii.tgz', 'xml.tgz']

        for filename in files_to_download:
            filepath = self.raw_dir / filename
            extracted_marker = self.raw_dir / f'.{filename}.extracted'

            if filepath.exists() and extracted_marker.exists() and not force:
                print(f"✓ {filename} already downloaded and extracted")
                continue

            print(f"Downloading {filename}...")
            self._download_file(filename, filepath)

            print(f"Extracting {filename}...")
            self._extract_archive(filepath)
            extracted_marker.touch()

        print("✓ All IAM files downloaded and extracted")

    def _download_file(self, filename: str, dest_path: Path) -> None:
        """Download file from Hugging Face dataset."""
        print(f"  Fetching {filename} from {self.repo_id}...")

        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            token=self.hf_token,
            cache_dir=None,  # Use default HF cache
        )

        # Copy from HF cache to our raw directory
        shutil.copy(downloaded_path, dest_path)
        print(f"  ✓ Saved to {dest_path}")

    def _extract_archive(self, filepath: Path) -> None:
        """Extract .tgz archive to appropriate subdirectory."""
        import tarfile

        # Determine target directory based on filename
        filename = filepath.name
        if 'xml' in filename:
            target_dir = self.raw_dir / 'xml'
        elif 'ascii' in filename:
            target_dir = self.raw_dir / 'ascii'
        elif 'lines' in filename:
            target_dir = self.raw_dir / 'lines'
        else:
            target_dir = self.raw_dir

        target_dir.mkdir(exist_ok=True)

        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=target_dir)

    def process(self) -> None:
        """
        Process raw IAM data into training-ready format:
        - Parse XML ground truth
        - Organize line images
        - Create train/val/test splits
        """
        print("Processing IAM data...")

        # Create output structure
        lines_dir = self.processed_dir / 'lines'
        gt_dir = self.processed_dir / 'ground_truth'
        splits_dir = self.processed_dir / 'splits'

        lines_dir.mkdir(exist_ok=True)
        gt_dir.mkdir(exist_ok=True)
        splits_dir.mkdir(exist_ok=True)

        # Parse XML annotations
        xml_dir = self.raw_dir / 'xml'
        lines_data = self._parse_xml_annotations(xml_dir)

        # Copy line images and create ground truth files
        print("Organizing line images and ground truth...")
        valid_lines = []

        for line_id, data in tqdm(lines_data.items()):
            # Find source image
            src_img = self._find_line_image(line_id)
            if src_img is None:
                continue

            # Copy image
            dst_img = lines_dir / f'{line_id}.png'
            shutil.copy(src_img, dst_img)

            # Write ground truth
            gt_file = gt_dir / f'{line_id}.txt'
            with open(gt_file, 'w') as f:
                f.write(data['text'])

            valid_lines.append({
                'line_id': line_id,
                'writer_id': data['writer_id'],
                'text': data['text']
            })

        # Create train/val/test splits (by writer)
        self._create_splits(valid_lines, splits_dir)

        print(f"✓ Processed {len(valid_lines)} lines")
        print(f"✓ Data ready at {self.processed_dir}")

    def _parse_xml_annotations(self, xml_dir: Path) -> Dict[str, Dict]:
        """Parse IAM XML files to extract line-level ground truth."""
        lines_data = {}

        for xml_file in xml_dir.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            writer_id = root.get('writer-id')

            for line in root.iter('line'):
                line_id = line.get('id')
                text = line.get('text', '')

                # Skip lines marked as segmentation errors
                if text == '' or text.startswith('#'):
                    continue

                lines_data[line_id] = {
                    'text': text,
                    'writer_id': writer_id
                }

        return lines_data

    def _find_line_image(self, line_id: str) -> Optional[Path]:
        """Locate line image in raw data structure."""
        # IAM structure: lines/a01/a01-000u/a01-000u-00.png
        parts = line_id.split('-')
        if len(parts) < 3:
            return None

        form_id = f"{parts[0]}-{parts[1]}"
        subdir = self.raw_dir / 'lines' / parts[0] / form_id

        img_path = subdir / f'{line_id}.png'
        return img_path if img_path.exists() else None

    def _create_splits(self, lines: List[Dict], splits_dir: Path) -> None:
        """
        Create train/val/test splits by writer ID.
        Standard IAM splits: ~80% train, ~10% val, ~10% test
        """
        # Group by writer
        writers = {}
        for line in lines:
            writer_id = line['writer_id']
            if writer_id not in writers:
                writers[writer_id] = []
            writers[writer_id].append(line['line_id'])

        # Split writers
        writer_ids = sorted(writers.keys())
        np.random.seed(42)
        np.random.shuffle(writer_ids)

        n_writers = len(writer_ids)
        n_train = int(0.8 * n_writers)
        n_val = int(0.1 * n_writers)

        train_writers = set(writer_ids[:n_train])
        val_writers = set(writer_ids[n_train:n_train + n_val])
        test_writers = set(writer_ids[n_train + n_val:])

        # Write splits
        splits = {'train': [], 'val': [], 'test': []}
        for line in lines:
            writer_id = line['writer_id']
            if writer_id in train_writers:
                splits['train'].append(line['line_id'])
            elif writer_id in val_writers:
                splits['val'].append(line['line_id'])
            else:
                splits['test'].append(line['line_id'])

        for split_name, line_ids in splits.items():
            split_file = splits_dir / f'{split_name}.txt'
            with open(split_file, 'w') as f:
                f.write('\n'.join(line_ids))

        print(f"  Train: {len(splits['train'])} lines")
        print(f"  Val: {len(splits['val'])} lines")
        print(f"  Test: {len(splits['test'])} lines")


class PacketProcessor:
    """
    Processes user personalization packets:
    1. Detect QR codes at corners for alignment
    2. Apply perspective transform to de-skew
    3. Segment and crop individual lines
    4. Pair with ground truth text

    Args:
        user_dir: Directory to store user's processed data
    """

    def __init__(self, user_dir: str):
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self.lines_dir = self.user_dir / 'lines'
        self.lines_dir.mkdir(exist_ok=True)

    def process_packet(
        self,
        image_paths: List[str],
        ground_truth: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Process photographed packet pages.

        Args:
            image_paths: List of paths to photographed pages
            ground_truth: Dict mapping line_id -> expected text

        Returns:
            Processing results with detected line count and quality metrics
        """
        all_lines = []

        for img_path in tqdm(image_paths, desc="Processing packet pages"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Detect QR codes and de-skew
            aligned_img = self._detect_and_align(img)
            if aligned_img is None:
                print(f"Warning: Could not align {img_path}")
                # Fall back to original image
                aligned_img = img

            # Segment lines
            line_imgs = self._segment_lines(aligned_img)
            all_lines.extend(line_imgs)

        # Save cropped lines
        saved_lines = []
        for i, line_img in enumerate(all_lines):
            line_id = f"line_{i:03d}"
            line_path = self.lines_dir / f'{line_id}.png'
            cv2.imwrite(str(line_path), line_img)
            saved_lines.append(line_id)

        # Match with ground truth
        matched_gt = self._match_ground_truth(saved_lines, ground_truth)

        # Save ground truth mapping
        gt_file = self.user_dir / 'ground_truth.json'
        with open(gt_file, 'w') as f:
            json.dump(matched_gt, f, indent=2)

        return {
            'lines_detected': len(saved_lines),
            'lines_matched': len(matched_gt),
            'output_dir': str(self.user_dir)
        }

    def _detect_and_align(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect QR codes at corners and apply perspective transform.

        Expected QR layout:
        TL -------- TR
        |           |
        |           |
        BL -------- BR
        """
        # Detect all QR codes
        qr_codes = pyzbar.decode(img)

        if len(qr_codes) < 4:
            # Not enough QR codes for alignment
            return None

        # Parse QR codes to identify corners
        corners = {}
        for qr in qr_codes:
            data = qr.data.decode('utf-8')
            # Expected format: "TL", "TR", "BL", "BR"
            if data in ['TL', 'TR', 'BL', 'BR']:
                # Get center of QR code
                rect = qr.rect
                center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
                corners[data] = center

        if len(corners) != 4:
            return None

        # Define source and destination points
        src_pts = np.float32([
            corners['TL'],
            corners['TR'],
            corners['BR'],
            corners['BL']
        ])

        # Calculate destination dimensions
        width = max(
            np.linalg.norm(np.array(corners['TR']) - np.array(corners['TL'])),
            np.linalg.norm(np.array(corners['BR']) - np.array(corners['BL']))
        )
        height = max(
            np.linalg.norm(np.array(corners['BL']) - np.array(corners['TL'])),
            np.linalg.norm(np.array(corners['BR']) - np.array(corners['TR']))
        )

        dst_pts = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned = cv2.warpPerspective(img, matrix, (int(width), int(height)))

        return aligned

    def _segment_lines(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Segment image into individual text lines.

        Uses horizontal projection to detect line boundaries.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection
        h_projection = np.sum(binary, axis=1)

        # Find line boundaries (regions with significant ink)
        threshold = np.mean(h_projection) * 0.3
        in_line = h_projection > threshold

        # Detect transitions
        lines = []
        line_start = None

        for i, is_line in enumerate(in_line):
            if is_line and line_start is None:
                line_start = i
            elif not is_line and line_start is not None:
                # Add some padding
                y1 = max(0, line_start - 5)
                y2 = min(img.shape[0], i + 5)

                # Crop line
                line_img = img[y1:y2, :]

                # Filter out very small lines (noise)
                if line_img.shape[0] > 10:
                    lines.append(line_img)

                line_start = None

        # Handle last line
        if line_start is not None:
            y1 = max(0, line_start - 5)
            line_img = img[y1:, :]
            if line_img.shape[0] > 10:
                lines.append(line_img)

        return lines

    def _match_ground_truth(
        self,
        line_ids: List[str],
        ground_truth: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Match detected lines with ground truth text.

        For now, assumes sequential order (line_000 -> first ground truth, etc.)
        More sophisticated matching could use OCR confidence or edit distance.
        """
        matched = {}
        gt_list = list(ground_truth.values())

        for i, line_id in enumerate(line_ids):
            if i < len(gt_list):
                matched[line_id] = gt_list[i]

        return matched
