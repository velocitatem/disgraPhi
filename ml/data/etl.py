"""
ETL pipeline for DisgraPhi data processing.

Components:
- IAMDownloader: Download and cache IAM Handwriting Database
- PacketProcessor: Process user packets (QR detection, de-skew, line crop)
"""

import os
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    1. Detect QR codes or ArUco markers at corners for alignment
    2. Apply perspective transform to de-skew
    3. Normalize lighting via Sauvola adaptive thresholding
    4. Segment and crop individual lines
    5. Pair with ground truth text

    Args:
        user_dir: Directory to store user's processed data
        use_aruco: Prefer ArUco marker detection over QR codes
        normalize_lighting: Apply Sauvola binarization for shadow removal
    """

    def __init__(self, user_dir: str, use_aruco: bool = False, normalize_lighting: bool = True):
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.use_aruco = use_aruco
        self.normalize_lighting = normalize_lighting

        self.lines_dir = self.user_dir / 'lines'
        self.lines_dir.mkdir(exist_ok=True)
        self.page_history: List[Dict[str, Any]] = []

    def process_packet(
        self,
        image_paths: List[str],
        ground_truth: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Process photographed packet pages.

        Args:
            image_paths: List of paths to photographed pages
            ground_truth: Dict mapping line_id -> expected text

        Returns:
            Processing results with detected line count and quality metrics
        """
        all_lines = []
        self.page_history = []

        for img_path in tqdm(image_paths, desc="Processing packet pages"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Detect QR codes / ArUco markers and de-skew
            if self.use_aruco:
                alignment = self._detect_aruco_and_align(img)
            else:
                alignment = self._detect_and_align(img)

            if alignment is None:
                print(f"Warning: Could not align {img_path} (no QR codes detected)")
                aligned_img = img
                page_context = None
                page_votes: Dict[str, int] = {}
            else:
                aligned_img = alignment.get('image')
                page_context = alignment.get('page')
                page_votes = alignment.get('votes', {})

                if aligned_img is None:
                    print(
                        f"Warning: Could not align {img_path} (insufficient corner detections)"
                    )
                    aligned_img = img

            self.page_history.append(
                {
                    "image": str(img_path),
                    "page": page_context,
                    "votes": page_votes,
                }
            )

            if page_context:
                page_id = page_context.get('id')
                page_num = page_context.get('number')
                page_total = page_context.get('total')
                if page_num is not None and page_total is not None:
                    page_label = f"{page_num}/{page_total}"
                else:
                    page_label = page_num if page_num is not None else page_id
                print(
                    f"Detected packet page {page_label} (id {page_id}) for {img_path}"
                )

            # Normalize lighting (Sauvola binarization for shadow removal)
            if self.normalize_lighting:
                aligned_img = self._normalize_lighting(aligned_img)

            # Extract textarea rectangles
            textarea_imgs = self._extract_textareas(aligned_img)
            all_lines.extend(textarea_imgs)

        # Save cropped textareas
        saved_textareas = []
        for i, textarea_img in enumerate(all_lines):
            textarea_id = f"textarea_{i:03d}"
            textarea_path = self.lines_dir / f'{textarea_id}.png'
            cv2.imwrite(str(textarea_path), textarea_img)
            saved_textareas.append(textarea_id)

        # Match with ground truth
        matched_gt = self._match_ground_truth(saved_textareas, ground_truth)

        # Save ground truth mapping
        gt_file = self.user_dir / 'ground_truth.json'
        with open(gt_file, 'w') as f:
            json.dump(matched_gt, f, indent=2)

        return {
            'textareas_detected': len(saved_textareas),
            'textareas_matched': len(matched_gt),
            'output_dir': str(self.user_dir),
            'page_history': self.page_history,
        }

    def _detect_and_align(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect QR codes at corners, align the page, and recover embedded metadata.

        Returns a dictionary with the aligned image (if possible), page metadata,
        and voting statistics derived from the detected QR payloads. Returns None
        when no QR payloads can be interpreted.
        """
        qr_codes = pyzbar.decode(img)

        corners: Dict[str, tuple[int, int]] = {}
        payloads: List[Dict[str, Any]] = []
        page_votes: Counter[str] = Counter()
        page_context: Optional[Dict[str, Any]] = None

        for qr in qr_codes:
            raw_data = qr.data.decode('utf-8')
            payload = self._parse_qr_payload(raw_data)
            if payload is None:
                continue

            payload.setdefault('raw', raw_data)

            corner_id = payload.get('corner')
            if corner_id in ['TL', 'TR', 'BL', 'BR']:
                rect = qr.rect
                center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
                corners[corner_id] = center

            page_data = payload.get('page')
            page_id: Optional[str] = None

            if isinstance(page_data, dict):
                page_id = page_data.get('id')
                if not page_id:
                    # Support older payloads with flattened id field
                    page_id = page_data.get('page')
            elif isinstance(page_data, str):
                page_id = page_data

            if page_id:
                payload['page_id'] = page_id
                page_votes[page_id] += 1

            payloads.append(payload)

        if page_votes:
            best_page_id, _ = page_votes.most_common(1)[0]
            for payload in payloads:
                if payload.get('page_id') == best_page_id:
                    page_data = payload.get('page')
                    context: Dict[str, Any] = {'id': best_page_id}
                    if isinstance(page_data, dict):
                        context.update(
                            {
                                'number': page_data.get('number'),
                                'total': page_data.get('total'),
                                'entries': page_data.get('entries'),
                            }
                        )
                    page_context = context
                    break

        aligned = None
        if len(corners) == 4:
            src_pts = np.float32([
                corners['TL'],
                corners['TR'],
                corners['BR'],
                corners['BL']
            ])

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

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            aligned = cv2.warpPerspective(img, matrix, (int(width), int(height)))

        if not payloads:
            return None

        return {
            'image': aligned,
            'page': page_context,
            'votes': dict(page_votes),
            'corners': list(corners.keys()),
            'payloads': payloads,
        }

    def _parse_qr_payload(self, data: str) -> Optional[Dict[str, Any]]:
        """Parse a QR payload supporting both legacy corner labels and JSON blobs."""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        if data in ['TL', 'TR', 'BL', 'BR']:
            return {'corner': data}

        return None

    def _detect_aruco_and_align(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect ArUco markers at page corners and apply perspective correction.

        ArUco IDs map to corners: 0=TL, 1=TR, 2=BR, 3=BL.
        Returns dict with aligned image or None if no markers found.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Fallback for older OpenCV versions
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            params = cv2.aruco.DetectorParameters_create()
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if marker_ids is None or len(marker_ids) == 0:
            return None

        id_to_corner = {0: 'TL', 1: 'TR', 2: 'BR', 3: 'BL'}
        corners: Dict[str, tuple] = {}

        for i, mid in enumerate(marker_ids.flatten()):
            corner_label = id_to_corner.get(int(mid))
            if corner_label:
                c = marker_corners[i][0]
                center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                corners[corner_label] = center

        aligned = None
        if len(corners) == 4:
            src_pts = np.float32([corners['TL'], corners['TR'], corners['BR'], corners['BL']])
            width = max(
                np.linalg.norm(np.array(corners['TR']) - np.array(corners['TL'])),
                np.linalg.norm(np.array(corners['BR']) - np.array(corners['BL']))
            )
            height = max(
                np.linalg.norm(np.array(corners['BL']) - np.array(corners['TL'])),
                np.linalg.norm(np.array(corners['BR']) - np.array(corners['TR']))
            )
            dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            aligned = cv2.warpPerspective(img, matrix, (int(width), int(height)))

        return {
            'image': aligned,
            'page': None,
            'votes': {},
            'corners': list(corners.keys()),
            'payloads': [],
        }

    def _normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize lighting using Sauvola adaptive thresholding.

        Strips out shadows, uneven illumination, and varying paper colors,
        leaving high-contrast ink strokes on a clean white background.
        Returns a 3-channel image suitable for downstream processing.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Sauvola binarization: threshold = mean * (1 + k * (std/R - 1))
        # where R = max(std) = 128 for uint8
        window_size = 25
        k = 0.2
        R = 128.0

        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        mean_sq = cv2.blur((gray.astype(np.float64)) ** 2, (window_size, window_size))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        threshold = mean * (1.0 + k * (std / R - 1.0))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

        # Return as 3-channel for consistency with downstream pipeline
        if len(img.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary

    def _extract_textareas(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Extract complete textarea rectangles from the aligned page.

        Uses horizontal projection to detect content blocks (handwritten areas)
        and extracts them as complete rectangles rather than individual lines.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize to detect ink
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use horizontal projection to find content blocks
        h_projection = np.sum(binary, axis=1)

        # Find regions with significant ink (handwritten content)
        # Use a lower threshold to capture sparse writing
        threshold = np.mean(h_projection) * 0.1
        in_content = h_projection > threshold

        # Find content blocks
        textareas = []
        img_height, img_width = img.shape[:2]
        block_start = None
        min_block_height = 50  # Minimum height for a valid textarea

        for i, has_content in enumerate(in_content):
            if has_content and block_start is None:
                block_start = i
            elif not has_content and block_start is not None:
                # Check if we have a large enough gap to consider this a separate block
                block_height = i - block_start

                if block_height > min_block_height:
                    # Extract the entire width of the content block
                    # Add padding around the detected content
                    padding_y = 15
                    padding_x = 20

                    y1 = max(0, block_start - padding_y)
                    y2 = min(img_height, i + padding_y)
                    x1 = padding_x  # Use small padding from left edge
                    x2 = img_width - padding_x  # Use small padding from right edge

                    textarea_img = img[y1:y2, x1:x2]
                    textareas.append((block_start, textarea_img))

                block_start = None

        # Handle last block if it extends to the end
        if block_start is not None:
            block_height = img_height - block_start
            if block_height > min_block_height:
                padding_y = 15
                padding_x = 20

                y1 = max(0, block_start - padding_y)
                y2 = img_height
                x1 = padding_x
                x2 = img_width - padding_x

                textarea_img = img[y1:y2, x1:x2]
                textareas.append((block_start, textarea_img))

        # Sort textareas by vertical position (top to bottom)
        textareas.sort(key=lambda t: t[0])

        # Return just the images, without coordinates
        return [textarea for _, textarea in textareas]

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
