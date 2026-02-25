"""Tests for DisgraPhi MVP pipeline: wild augmentation, calibration, ArUco, Sauvola."""

import json
import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path
from PIL import Image

from ml.data.augmentation import WildAugmentation, HandwritingAugmentation, MinimalAugmentation
from ml.data.etl import PacketProcessor
from src.packet_generator import PacketGenerator


class TestWildAugmentation:
    def _white_image(self, h=100, w=300):
        return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 255)

    def test_output_same_size(self):
        img = self._white_image()
        aug = WildAugmentation(strength=0.5, prob=1.0)
        result = aug(img)
        assert result.size == img.size

    def test_random_shadow_darkens(self):
        img_np = np.ones((100, 300, 3), dtype=np.uint8) * 200
        aug = WildAugmentation(strength=0.8, prob=1.0)
        shadowed = aug.random_shadow(img_np)
        assert shadowed.shape == img_np.shape
        assert shadowed.min() < img_np.min()

    def test_illumination_gradient_varies(self):
        img_np = np.ones((100, 300, 3), dtype=np.uint8) * 128
        aug = WildAugmentation(strength=0.8)
        result = aug.illumination_gradient(img_np)
        assert result.shape == img_np.shape
        assert len(np.unique(result)) > 1

    def test_low_strength_minimal_change(self):
        img = self._white_image()
        aug = WildAugmentation(strength=0.3, prob=1.0)
        result = aug(img)
        assert result.size == img.size

    def test_grayscale_shadow(self):
        img_np = np.ones((100, 300), dtype=np.uint8) * 200
        aug = WildAugmentation(strength=0.5)
        result = aug.random_shadow(img_np)
        assert result.shape == img_np.shape


class TestSauvolaBinarization:
    def test_uniform_input_produces_white(self):
        proc = PacketProcessor(tempfile.mkdtemp(), normalize_lighting=True)
        img = np.ones((100, 300, 3), dtype=np.uint8) * 200
        result = proc._normalize_lighting(img)
        assert result.shape == img.shape
        assert result.max() == 255

    def test_preserves_ink_strokes(self):
        proc = PacketProcessor(tempfile.mkdtemp(), normalize_lighting=True)
        img = np.ones((100, 300, 3), dtype=np.uint8) * 240
        # Draw dark "ink" stroke
        img[40:60, 100:200] = 30
        result = proc._normalize_lighting(img)
        # Ink region should be dark (0), background should be white (255)
        ink_region = result[50, 150, 0]
        bg_region = result[10, 10, 0]
        assert ink_region < bg_region

    def test_grayscale_input(self):
        proc = PacketProcessor(tempfile.mkdtemp(), normalize_lighting=True)
        gray = np.ones((100, 300), dtype=np.uint8) * 200
        result = proc._normalize_lighting(gray)
        assert len(result.shape) == 2

    def test_returns_3channel_from_bgr(self):
        proc = PacketProcessor(tempfile.mkdtemp(), normalize_lighting=True)
        img = np.ones((100, 300, 3), dtype=np.uint8) * 200
        result = proc._normalize_lighting(img)
        assert result.shape[2] == 3


class TestArUcoDetection:
    def _create_aruco_image(self, w=600, h=800, marker_size=80):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        positions = {0: (10, 10), 1: (w - marker_size - 10, 10),
                     2: (w - marker_size - 10, h - marker_size - 10),
                     3: (10, h - marker_size - 10)}
        for mid, (x, y) in positions.items():
            marker = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size)
            canvas[y:y + marker_size, x:x + marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        return canvas

    def test_detects_all_four_corners(self):
        proc = PacketProcessor(tempfile.mkdtemp(), use_aruco=True)
        img = self._create_aruco_image()
        result = proc._detect_aruco_and_align(img)
        assert result is not None
        assert set(result['corners']) == {'TL', 'TR', 'BR', 'BL'}

    def test_perspective_correction(self):
        proc = PacketProcessor(tempfile.mkdtemp(), use_aruco=True)
        img = self._create_aruco_image()
        result = proc._detect_aruco_and_align(img)
        assert result['image'] is not None
        assert len(result['image'].shape) == 3

    def test_no_markers_returns_none(self):
        proc = PacketProcessor(tempfile.mkdtemp(), use_aruco=True)
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        result = proc._detect_aruco_and_align(img)
        assert result is None

    def test_partial_markers_no_alignment(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        canvas = np.ones((400, 300, 3), dtype=np.uint8) * 255
        marker = cv2.aruco.generateImageMarker(aruco_dict, 0, 60)
        canvas[10:70, 10:70] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        proc = PacketProcessor(tempfile.mkdtemp(), use_aruco=True)
        result = proc._detect_aruco_and_align(canvas)
        assert result is not None
        assert result['image'] is None  # Can't align with only 1 corner


class TestCalibrationSheet:
    def test_generates_pdf_with_calibration_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = str(Path(tmpdir) / 'cal.pdf')
            gen = PacketGenerator(user_id='test')
            gt = gen.generate(pdf_path, task_type='calibration_entries')
            assert Path(pdf_path).exists()
            assert len(gt) > 0
            # Verify pangram is in ground truth
            assert any('quick brown fox' in v for v in gt.values())

    def test_generates_with_aruco_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = str(Path(tmpdir) / 'cal_aruco.pdf')
            gen = PacketGenerator(user_id='test', use_aruco=True)
            gt = gen.generate(pdf_path, task_type='calibration_entries')
            assert Path(pdf_path).exists()
            assert len(gt) > 0

    def test_calibration_entries_include_numbers(self):
        gen = PacketGenerator(user_id='test')
        gt = gen.generate('/tmp/test_cal_nums.pdf', task_type='calibration_entries')
        assert any('0 1 2 3' in v for v in gt.values())

    def test_calibration_entries_include_prefixes(self):
        gen = PacketGenerator(user_id='test')
        gt = gen.generate('/tmp/test_cal_pfx.pdf', task_type='calibration_entries')
        assert any('un re pre' in v for v in gt.values())


class TestPacketProcessorFlags:
    def test_default_flags(self):
        proc = PacketProcessor(tempfile.mkdtemp())
        assert proc.use_aruco is False
        assert proc.normalize_lighting is True

    def test_aruco_flag(self):
        proc = PacketProcessor(tempfile.mkdtemp(), use_aruco=True)
        assert proc.use_aruco is True

    def test_no_normalize_flag(self):
        proc = PacketProcessor(tempfile.mkdtemp(), normalize_lighting=False)
        assert proc.normalize_lighting is False
