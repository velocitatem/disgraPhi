"""
Model Providers for DisgraPhi
pip install einops timm

Unified interface for different vision-language models:
- Qwen3-VL (2B/4B/7B)
- SmolVLM (256M/500M/2.2B)
- Florence-2 (Base/Large/Base-FT/Large-FT)
- Future models...

Each provider exposes the same interface for training and inference.
"""

from .base import BaseVisionLanguageModel
from .qwen3 import Qwen3VLModel
from .smolvlm import SmolVLMModel
from .florence2 import Florence2Model

# Model registry for easy instantiation
MODEL_REGISTRY = {
    'qwen3-vl-2b': Qwen3VLModel,
    'qwen3-vl-4b': Qwen3VLModel,
    'qwen3-vl-7b': Qwen3VLModel,
    'smolvlm-256m': SmolVLMModel,
    'smolvlm-500m': SmolVLMModel,
    'smolvlm-2.2b': SmolVLMModel,
    'florence2-base': Florence2Model,
    'florence2-large': Florence2Model,
    'florence2-base-ft': Florence2Model,
    'florence2-large-ft': Florence2Model,
}


def create_model(
    model_type: str,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Factory function to create model instances.

    Args:
        model_type: Model identifier (e.g., 'qwen3-vl-4b', 'smolvlm-256m')
        **kwargs: Model-specific arguments

    Returns:
        Model instance implementing BaseVisionLanguageModel interface

    Example:
        >>> model = create_model('smolvlm-256m', load_in_4bit=True)
        >>> model = create_model('qwen3-vl-4b', lora_r=16)
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}\n"
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_type]
    return model_class(model_type=model_type, **kwargs)


__all__ = [
    'BaseVisionLanguageModel',
    'Qwen3VLModel',
    'SmolVLMModel',
    'Florence2Model',
    'MODEL_REGISTRY',
    'create_model'
]
