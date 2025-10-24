"""
Base interface for vision-language models.

All model providers must implement this interface to ensure
compatibility with the training loop and evaluation pipeline.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path


class BaseVisionLanguageModel(nn.Module, ABC):
    """
    Abstract base class for vision-language models.

    All providers (Qwen3, SmolVLM, etc.) must implement this interface.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.tokenizer = None

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Image tensor
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Target token IDs for training
            **kwargs: Model-specific arguments

        Returns:
            Dictionary with at least:
            - loss: Training loss (if labels provided)
            - logits: Model logits
        """
        pass

    @abstractmethod
    def generate(
        self,
        pixel_values: Any,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """
        Generate text from image + prompt.

        Args:
            pixel_values: Image(s) - can be tensor, PIL Image, or path
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Model-specific generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def save_adapter(self, output_dir: str) -> None:
        """
        Save model adapter/weights.

        Args:
            output_dir: Directory to save to
        """
        pass

    @abstractmethod
    def load_adapter(self, adapter_path: str) -> None:
        """
        Load model adapter/weights.

        Args:
            adapter_path: Path to adapter
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> int:
        """
        Get count of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        pass

    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        percentage = 100 * trainable / total if total > 0 else 0

        print(f"Trainable params: {trainable:,} || "
              f"Total params: {total:,} || "
              f"Trainable%: {percentage:.2f}%")

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.parameters()).dtype
