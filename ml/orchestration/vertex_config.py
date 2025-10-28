"""
Configuration classes for Vertex AI training jobs.

This module defines configuration structures for running training jobs
on GCP Vertex AI, including hyperparameter tuning specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ParameterType(str, Enum):
    """Types of hyperparameters for tuning."""
    DOUBLE = "DOUBLE"
    INTEGER = "INTEGER"
    CATEGORICAL = "CATEGORICAL"
    DISCRETE = "DISCRETE"


class ScaleType(str, Enum):
    """Scale types for hyperparameter search."""
    UNIT_LINEAR_SCALE = "UNIT_LINEAR_SCALE"
    UNIT_LOG_SCALE = "UNIT_LOG_SCALE"
    UNIT_REVERSE_LOG_SCALE = "UNIT_REVERSE_LOG_SCALE"


@dataclass
class HyperparameterSpec:
    """
    Specification for a single hyperparameter.
    
    This maps to the training script's command-line arguments.
    """
    parameter_id: str  # Command-line argument name (e.g., "lora_r", "learning_rate")
    parameter_type: ParameterType
    
    # For DOUBLE and INTEGER types
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    scale_type: ScaleType = ScaleType.UNIT_LINEAR_SCALE
    
    # For CATEGORICAL and DISCRETE types
    categorical_values: Optional[List[str]] = None
    discrete_values: Optional[List[Union[int, float]]] = None
    
    def to_vertex_dict(self) -> Dict[str, Any]:
        """Convert to Vertex AI API format."""
        spec = {
            "parameterId": self.parameter_id,
            "parameterType": self.parameter_type.value,
        }
        
        if self.parameter_type in [ParameterType.DOUBLE, ParameterType.INTEGER]:
            spec["minValue"] = self.min_value
            spec["maxValue"] = self.max_value
            spec["scaleType"] = self.scale_type.value
        elif self.parameter_type == ParameterType.CATEGORICAL:
            spec["categoricalValues"] = self.categorical_values
        elif self.parameter_type == ParameterType.DISCRETE:
            spec["discreteValues"] = self.discrete_values
            
        return spec


@dataclass
class VertexAIConfig:
    """
    Configuration for Vertex AI training jobs.
    
    This class encapsulates all settings needed to run training on Vertex AI,
    while keeping the core training code unchanged.
    """
    
    # GCP Project settings
    project_id: str
    region: str = "us-central1"
    staging_bucket: str = None  # Will use gs://{project_id}-vertex-staging if None
    
    # Training job settings
    display_name: Optional[str] = None  # Auto-generated if None
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    
    # Container settings
    container_uri: str = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
    python_module: str = "ml.models.train"  # Entry point module
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    hyperparameter_specs: List[HyperparameterSpec] = field(default_factory=list)
    max_trial_count: int = 10
    parallel_trial_count: int = 2
    metric_id: str = "eval_loss"
    metric_goal: str = "MINIMIZE"  # MINIMIZE or MAXIMIZE
    
    # Training script arguments (passed to train.py)
    base_args: Dict[str, Any] = field(default_factory=dict)
    
    # Dataset and output paths (GCS paths)
    data_dir_gcs: Optional[str] = None
    output_dir_gcs: Optional[str] = None
    
    # Service account for Vertex AI jobs
    service_account: Optional[str] = None
    
    # Network configuration
    network: Optional[str] = None
    
    # Timeout
    timeout_seconds: int = 7200  # 2 hours default
    
    def __post_init__(self):
        """Set default values."""
        if self.staging_bucket is None:
            self.staging_bucket = f"gs://{self.project_id}-vertex-staging"
            
        if self.display_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tuning_suffix = "_tuning" if self.enable_hyperparameter_tuning else ""
            self.display_name = f"disgraphi_train{tuning_suffix}_{timestamp}"
    
    def get_worker_pool_spec(self) -> Dict[str, Any]:
        """Generate worker pool specification for Vertex AI."""
        spec = {
            "machineSpec": {
                "machineType": self.machine_type,
            },
            "replicaCount": 1,
            "containerSpec": {
                "imageUri": self.container_uri,
                "pythonPackageSpec": {
                    "executorImageUri": self.container_uri,
                    "packageUris": [],  # Will be populated with training code
                    "pythonModule": self.python_module,
                    "args": self._build_training_args(),
                }
            }
        }
        
        if self.accelerator_type and self.accelerator_count > 0:
            spec["machineSpec"]["acceleratorType"] = self.accelerator_type
            spec["machineSpec"]["acceleratorCount"] = self.accelerator_count
            
        return spec
    
    def _build_training_args(self) -> List[str]:
        """Build command-line arguments for train.py."""
        args = []
        for key, value in self.base_args.items():
            args.append(f"--{key}")
            if value is not True:  # Don't add value for boolean flags
                args.append(str(value))
        return args
    
    def get_hyperparameter_tuning_config(self) -> Optional[Dict[str, Any]]:
        """Generate hyperparameter tuning configuration."""
        if not self.enable_hyperparameter_tuning:
            return None
            
        return {
            "studySpec": {
                "metrics": [{
                    "metricId": self.metric_id,
                    "goal": self.metric_goal
                }],
                "parameters": [
                    spec.to_vertex_dict() for spec in self.hyperparameter_specs
                ],
            },
            "maxTrialCount": self.max_trial_count,
            "parallelTrialCount": self.parallel_trial_count,
        }
