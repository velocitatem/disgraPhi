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
    def prepare_training_batch(
        self,
        images: List[Any],
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Prepare a batch of images and texts for training.

        Each model handles its own preprocessing (chat templates, prompts, etc.).

        Args:
            images: List of PIL Images
            texts: List of ground truth texts

        Returns:
            Dictionary with processed inputs ready for forward pass:
            - pixel_values: Processed image tensors
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Training labels
            - image_grid_thw: (optional) Image grid dimensions for some models
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
    def load_adapter(
        self,
        adapter_path: str,
        adapter_name: str = "default",
        is_trainable: bool = True
    ) -> None:
        """
        Load model adapter/weights.

        Args:
            adapter_path: Path to adapter
            adapter_name: Name to assign to the adapter (for multi-adapter composition)
            is_trainable: Whether adapter parameters should be trainable
        """
        pass

    @abstractmethod
    def set_adapter(self, adapter_names: List[str]) -> None:
        """
        Set active adapters for inference/training.

        Args:
            adapter_names: List of adapter names to activate
        """
        pass

    @abstractmethod
    def get_adapter_state_dict(self, adapter_name: str = "default") -> Dict[str, Any]:
        """
        Get state dict for a specific adapter.

        Args:
            adapter_name: Name of the adapter

        Returns:
            Adapter state dict
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
