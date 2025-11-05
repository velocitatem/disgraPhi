"""
Handwriting-specific augmentation pipeline for few-shot personalization.

Strong but realistic augmentations to simulate handwriting variability:
- Elastic distortion (pen pressure, hand movement)
- Slant variation (writing angle)
- Baseline drift (tremor, uneven writing)
- Stroke width variation (pen pressure)
- Spacing variation (letter/word spacing)
- Perspective transform (page angle)
- Paper/ink noise (texture, bleeding)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from typing import Optional, Tuple
import random


class HandwritingAugmentation:
    """
    Augmentation pipeline for handwriting images.

    Args:
        strength: Augmentation intensity 0.0-1.0 (default: 0.7)
        prob: Probability of applying each augmentation (default: 0.5)
    """

    def __init__(self, strength: float = 0.7, prob: float = 0.5):
        self.strength = strength
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to handwriting image."""
        # Convert to numpy for CV operations
        img_np = np.array(image)

        # Apply augmentations with probability
        if random.random() < self.prob:
            img_np = self.deskew_rotation(img_np)

        if random.random() < self.prob:
            img_np = self.elastic_distortion(img_np)

        if random.random() < self.prob:
            img_np = self.slant_variation(img_np)

        if random.random() < self.prob:
            img_np = self.baseline_drift(img_np)

        if random.random() < self.prob:
            img_np = self.perspective_transform(img_np)

        # Convert back to PIL for PIL-based augmentations
        image = Image.fromarray(img_np)

        if random.random() < self.prob:
            image = self.gaussian_blur(image)

        if random.random() < self.prob:
            image = self.stroke_width_variation(image)

        if random.random() < self.prob:
            image = self.spacing_variation(image)

        if random.random() < self.prob:
            image = self.paper_ink_noise(image)

        if random.random() < self.prob:
            image = self.jpeg_compression(image)

        return image

    def deskew_rotation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply ±2° rotation to simulate slight page rotation.

        Recommended by DeepSeek-OCR research for handwriting.
        """
        h, w = img.shape[:2]

        # Random rotation angle (±2 degrees)
        max_angle = 2.0 * self.strength
        angle = random.uniform(-max_angle, max_angle)

        # Rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        return rotated

    def gaussian_blur(self, img: Image.Image) -> Image.Image:
        """
        Apply mild Gaussian blur (sigma=0.5-1.0).

        Simulates slight out-of-focus or camera blur.
        Recommended by DeepSeek-OCR research.
        """
        sigma = 0.5 + 0.5 * self.strength  # Range: 0.5 - 1.0
        blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return blurred

    def jpeg_compression(self, img: Image.Image) -> Image.Image:
        """
        Apply JPEG compression artifacts (quality=80-95).

        Simulates phone camera or scanner compression.
        Recommended by DeepSeek-OCR research.
        """
        import io

        # Random quality (80-95)
        min_quality = 80
        max_quality = 95
        quality = int(max_quality - (max_quality - min_quality) * self.strength * random.random())

        # Compress to JPEG and decompress
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        return compressed

    def elastic_distortion(self, img: np.ndarray) -> np.ndarray:
        """
        Apply elastic distortion to simulate pen pressure and hand movement.

        Uses grid-based displacement to warp the image realistically.
        """
        h, w = img.shape[:2]

        # Control distortion strength
        alpha = int(30 * self.strength)  # Displacement intensity
        sigma = int(5 * self.strength)   # Smoothness

        # Create random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha

        # Smooth the displacement fields (gaussian filter)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Remap image
        distorted = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return distorted

    def slant_variation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply slant variation (±10-15°) to simulate writing angle.
        """
        h, w = img.shape[:2]

        # Random slant angle
        max_angle = 15 * self.strength
        angle = random.uniform(-max_angle, max_angle)

        # Create affine transform for slant (shear)
        shear_factor = np.tan(np.radians(angle))
        M = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        # Apply transform
        slanted = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        return slanted

    def baseline_drift(self, img: np.ndarray) -> np.ndarray:
        """
        Apply baseline drift to simulate tremor/uneven writing.

        Creates a sinusoidal vertical displacement along the horizontal axis.
        """
        h, w = img.shape[:2]

        # Sinusoidal drift parameters
        amplitude = int(5 * self.strength)  # Vertical displacement
        frequency = random.uniform(0.5, 2.0)  # Wave frequency
        phase = random.uniform(0, 2 * np.pi)

        # Create displacement map
        x = np.arange(w)
        drift = amplitude * np.sin(2 * np.pi * frequency * x / w + phase)

        # Apply row-wise vertical shift
        result = np.zeros_like(img)
        for i in range(w):
            shift = int(drift[i])
            if shift > 0:
                result[:h-shift, i] = img[shift:, i]
            elif shift < 0:
                result[-shift:, i] = img[:h+shift, i]
            else:
                result[:, i] = img[:, i]

        return result

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to simulate page angle.
        """
        h, w = img.shape[:2]

        # Random perspective strength
        max_offset = int(20 * self.strength)

        # Source points (corners)
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Destination points (with random offsets)
        dst_points = np.float32([
            [random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
            [random.randint(0, max_offset), h - random.randint(0, max_offset)]
        ])

        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transform
        warped = cv2.warpPerspective(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        return warped

    def stroke_width_variation(self, img: Image.Image) -> Image.Image:
        """
        Vary stroke width to simulate pen pressure variation.

        Uses morphological operations (erosion/dilation).
        """
        # Convert to grayscale for morphology
        if img.mode != 'L':
            gray = img.convert('L')
        else:
            gray = img

        img_np = np.array(gray)

        # Random operation: dilate (thicken) or erode (thin)
        if random.random() < 0.5:
            # Dilate (thicken strokes)
            kernel_size = int(1 + 2 * self.strength)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            result = cv2.dilate(img_np, kernel, iterations=1)
        else:
            # Erode (thin strokes)
            kernel_size = int(1 + 1 * self.strength)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            result = cv2.erode(img_np, kernel, iterations=1)

        # Convert back to RGB if needed
        result_img = Image.fromarray(result)
        if img.mode == 'RGB':
            result_img = result_img.convert('RGB')

        return result_img

    def spacing_variation(self, img: Image.Image) -> Image.Image:
        """
        Vary horizontal spacing (stretch/compress).
        """
        w, h = img.size

        # Random scale factor
        scale = 1.0 + random.uniform(-0.15, 0.15) * self.strength
        new_w = int(w * scale)

        # Resize and crop/pad to original size
        resized = img.resize((new_w, h), Image.BILINEAR)

        if new_w > w:
            # Crop center
            left = (new_w - w) // 2
            resized = resized.crop((left, 0, left + w, h))
        else:
            # Pad
            result = Image.new(img.mode, (w, h), 255)  # White background
            left = (w - new_w) // 2
            result.paste(resized, (left, 0))
            resized = result

        return resized

    def paper_ink_noise(self, img: Image.Image) -> Image.Image:
        """
        Add paper texture and ink bleeding noise.
        """
        # Convert to numpy
        img_np = np.array(img)

        # Add gaussian noise
        noise_strength = 10 * self.strength
        noise = np.random.randn(*img_np.shape) * noise_strength
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        # Add slight blur (ink bleeding)
        if random.random() < 0.5:
            noisy_img = Image.fromarray(noisy)
            noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=0.5 * self.strength))
            noisy = np.array(noisy_img)

        # Adjust contrast slightly
        contrast_factor = 1.0 + random.uniform(-0.1, 0.1) * self.strength
        noisy_img = Image.fromarray(noisy)
        enhancer = ImageEnhance.Contrast(noisy_img)
        result = enhancer.enhance(contrast_factor)

        return result


class MinimalAugmentation:
    """
    Minimal augmentation for validation/testing or very few samples (k < 5).

    Only applies light geometric transforms without distortion.
    """

    def __init__(self, strength: float = 0.3):
        self.strength = strength

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply minimal augmentations."""
        # Light rotation
        if random.random() < 0.5:
            angle = random.uniform(-2, 2) * self.strength
            image = image.rotate(angle, fillcolor=255)

        # Light scaling
        if random.random() < 0.5:
            w, h = image.size
            scale = 1.0 + random.uniform(-0.05, 0.05) * self.strength
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.BILINEAR)

            # Crop/pad to original size
            if scale > 1.0:
                left = (new_size[0] - w) // 2
                top = (new_size[1] - h) // 2
                image = image.crop((left, top, left + w, top + h))
            else:
                result = Image.new('RGB', (w, h), 255)
                left = (w - new_size[0]) // 2
                top = (h - new_size[1]) // 2
                result.paste(image, (left, top))
                image = result

        return image


# Augmentation presets for different scenarios
AUGMENTATION_PRESETS = {
    'conservative': {
        'strength': 0.5,
        'prob': 0.4,
        'description': 'Light augmentation for clean handwriting or larger datasets (>100 samples)'
    },
    'standard': {
        'strength': 0.7,
        'prob': 0.5,
        'description': 'Balanced augmentation (recommended default for 50-100 samples)'
    },
    'aggressive': {
        'strength': 0.9,
        'prob': 0.6,
        'description': 'Heavy augmentation for very small datasets (<50 samples)'
    },
    'minimal': {
        'strength': 0.3,
        'prob': 0.3,
        'description': 'Minimal augmentation for validation or testing'
    }
}


def get_augmentation(preset: str = 'standard') -> HandwritingAugmentation:
    """
    Get augmentation pipeline by preset name.

    Args:
        preset: One of 'conservative', 'standard', 'aggressive', 'minimal'

    Returns:
        HandwritingAugmentation instance

    Example:
        >>> aug = get_augmentation('standard')
        >>> augmented_image = aug(image)
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Available: {list(AUGMENTATION_PRESETS.keys())}"
        )

    config = AUGMENTATION_PRESETS[preset]

    if preset == 'minimal':
        return MinimalAugmentation(strength=config['strength'])
    else:
        return HandwritingAugmentation(
            strength=config['strength'],
            prob=config['prob']
        )


def print_augmentation_presets():
    """Print available augmentation presets."""
    print("Available Augmentation Presets:")
    print("=" * 80)
    for name, config in AUGMENTATION_PRESETS.items():
        print(f"\n{name.upper()}")
        print(f"  Strength: {config['strength']}")
        print(f"  Probability: {config['prob']}")
        print(f"  Description: {config['description']}")
    print("=" * 80)


if __name__ == "__main__":
    # Demo augmentation presets
    print_augmentation_presets()

    # Test augmentation
    print("\nTesting augmentation...")
    from PIL import Image
    import numpy as np

    # Create dummy image
    img = Image.new('RGB', (200, 50), color='white')

    # Test each preset
    for preset_name in AUGMENTATION_PRESETS.keys():
        aug = get_augmentation(preset_name)
        augmented = aug(img)
        print(f"✓ {preset_name}: {augmented.size}")
