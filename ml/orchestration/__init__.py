"""
Orchestration module for running ML training on cloud platforms.

This module provides non-invasive hooks into the existing training pipeline
to enable execution on GCP Vertex AI and other cloud platforms while
maintaining compatibility with local development.
"""

from .vertex_config import (
    VertexAIConfig,
    HyperparameterSpec,
    ParameterType,
    ScaleType
)
from .vertex_launcher import VertexAILauncher

__all__ = [
    'VertexAIConfig',
    'HyperparameterSpec',
    'ParameterType',
    'ScaleType',
    'VertexAILauncher',
]
