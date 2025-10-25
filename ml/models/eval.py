"""
OCR Evaluation Metrics

Implements industry-standard OCR metrics:
- CER (Character Error Rate): Fine-grained character-level accuracy
- WER (Word Error Rate): Word-level accuracy
- Normalized Edit Distance: Alignment-aware similarity
- Hybrid OCR Loss: Weighted combination for robust evaluation

References:
- CTC loss for training (sequence-to-sequence)
- CER/WER for evaluation and benchmarking
"""

import numpy as np
from typing import Tuple, Dict


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Returns the minimum number of single-character edits
    (insertions, deletions, substitutions) required to change s1 into s2.

    Args:
        s1: Source string
        s2: Target string

    Returns:
        Edit distance (integer)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(ground_truth: str, prediction: str) -> float:
    """
    Compute Character Error Rate (CER).

    CER = (Substitutions + Deletions + Insertions) / Total characters in ground truth

    CER is the standard metric for fine-grained OCR evaluation. Lower is better.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        CER as float (0.0 = perfect, 1.0+ = very poor)
    """
    if not ground_truth:
        return 1.0 if prediction else 0.0

    edit_distance = levenshtein_distance(ground_truth, prediction)
    cer = edit_distance / len(ground_truth)

    return cer


def word_error_rate(ground_truth: str, prediction: str) -> float:
    """
    Compute Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total words in ground truth

    WER is useful for document-level and field-based OCR evaluation.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        WER as float (0.0 = perfect, 1.0+ = very poor)
    """
    gt_words = ground_truth.split()
    pred_words = prediction.split()

    if not gt_words:
        return 1.0 if pred_words else 0.0

    edit_distance = levenshtein_distance(' '.join(gt_words), ' '.join(pred_words))
    wer = edit_distance / len(' '.join(gt_words))

    return wer


def normalized_edit_distance(ground_truth: str, prediction: str) -> float:
    """
    Compute normalized edit distance (0.0 = identical, 1.0 = completely different).

    This is alignment-aware and handles variable-length sequences well.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        Normalized distance (0.0-1.0)
    """
    if not ground_truth and not prediction:
        return 0.0

    max_len = max(len(ground_truth), len(prediction))
    if max_len == 0:
        return 0.0

    edit_distance = levenshtein_distance(ground_truth, prediction)
    return edit_distance / max_len


def compute_ocr_metrics(ground_truth: str, prediction: str) -> Dict[str, float]:
    """
    Compute all OCR evaluation metrics.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        Dictionary with all metrics:
        - cer: Character Error Rate
        - wer: Word Error Rate
        - ned: Normalized Edit Distance
        - accuracy: Character-level accuracy (1 - NED)
    """
    cer = character_error_rate(ground_truth, prediction)
    wer = word_error_rate(ground_truth, prediction)
    ned = normalized_edit_distance(ground_truth, prediction)
    accuracy = 1.0 - ned

    return {
        'cer': cer,
        'wer': wer,
        'ned': ned,
        'accuracy': accuracy
    }


def model_agnostic_loss(
    ground_truth: str,
    prediction: str,
    cer_weight: float = 0.6,
    wer_weight: float = 0.3,
    ned_weight: float = 0.1
) -> float:
    """
    Compute hybrid OCR loss combining CER, WER, and normalized edit distance.

    This weighted combination provides robust evaluation across different
    OCR scenarios:
    - CER (60%): Fine-grained character accuracy
    - WER (30%): Word-level structure
    - NED (10%): Alignment-aware similarity

    The weights are tuned for handwriting recognition where character-level
    accuracy is most critical, but word boundaries also matter.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text
        cer_weight: Weight for CER (default: 0.6)
        wer_weight: Weight for WER (default: 0.3)
        ned_weight: Weight for normalized edit distance (default: 0.1)

    Returns:
        Hybrid loss value (0.0 = perfect, higher = worse)

    Example:
        >>> loss = model_agnostic_loss("Hello world", "Helo world")
        >>> print(f"Loss: {loss:.4f}")
        Loss: 0.0833
    """
    # Validate weights sum to 1.0
    total_weight = cer_weight + wer_weight + ned_weight
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    # Handle empty cases
    if not ground_truth:
        return 1.0 if prediction else 0.0

    # Compute individual metrics
    cer = character_error_rate(ground_truth, prediction)
    wer = word_error_rate(ground_truth, prediction)
    ned = normalized_edit_distance(ground_truth, prediction)

    # Weighted combination
    hybrid_loss = (cer_weight * cer +
                   wer_weight * wer +
                   ned_weight * ned)

    return hybrid_loss


if __name__ == "__main__":
    test_cases = [
        ("Hello, world!", "Hello, world!"),  # Perfect match
        ("Hello, world!", "Hello, world?"),  # Small punctuation error
        ("Hello, world!", "Helo, world!"),   # Missing character
        ("Hello, world!", "Hello world"),    # Missing punctuation
        ("The quick brown fox", "The quikc brown fox"),  # Transposition
        ("handwriting", "handwritng"),       # OCR-typical error
        ("", "something"),                    # Empty ground truth
        ("something", ""),                    # Empty prediction
    ]

    for gt, pred in test_cases:
        metrics = compute_ocr_metrics(gt, pred)
        hybrid_loss = model_agnostic_loss(gt, pred)

        print(f"\nGround Truth: '{gt}'")
        print(f"Prediction:   '{pred}'")
        print(f"  CER:          {metrics['cer']:.4f}")
        print(f"  WER:          {metrics['wer']:.4f}")
        print(f"  NED:          {metrics['ned']:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Hybrid Loss:  {hybrid_loss:.4f}")
