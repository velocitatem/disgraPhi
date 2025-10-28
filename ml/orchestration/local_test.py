"""
Local testing utilities for Vertex AI orchestration.

This module provides tools to test the orchestration setup locally
without requiring GCP credentials or actually submitting jobs.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from .vertex_config import VertexAIConfig, HyperparameterSpec


class LocalTester:
    """
    Test Vertex AI orchestration locally without GCP.
    
    This helps validate configuration and job specifications before
    submitting to Vertex AI.
    """
    
    def __init__(self, config: VertexAIConfig):
        """
        Initialize the local tester.
        
        Args:
            config: Vertex AI configuration to test
        """
        self.config = config
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not self.config.project_id:
            errors.append("project_id is required")
        
        if not self.config.region:
            errors.append("region is required")
        
        # Check hyperparameter tuning configuration
        if self.config.enable_hyperparameter_tuning:
            if not self.config.hyperparameter_specs:
                errors.append(
                    "enable_hyperparameter_tuning is True but no "
                    "hyperparameter_specs provided"
                )
            
            if self.config.max_trial_count < 1:
                errors.append("max_trial_count must be >= 1")
            
            if self.config.parallel_trial_count < 1:
                errors.append("parallel_trial_count must be >= 1")
            
            if self.config.parallel_trial_count > self.config.max_trial_count:
                errors.append(
                    "parallel_trial_count cannot exceed max_trial_count"
                )
        
        # Validate hyperparameter specs
        for i, spec in enumerate(self.config.hyperparameter_specs):
            spec_errors = self._validate_hyperparameter_spec(spec, i)
            errors.extend(spec_errors)
        
        return errors
    
    def _validate_hyperparameter_spec(
        self, spec: HyperparameterSpec, index: int
    ) -> List[str]:
        """Validate a single hyperparameter specification."""
        errors = []
        prefix = f"hyperparameter_specs[{index}]"
        
        if not spec.parameter_id:
            errors.append(f"{prefix}: parameter_id is required")
        
        from .vertex_config import ParameterType
        
        if spec.parameter_type in [ParameterType.DOUBLE, ParameterType.INTEGER]:
            if spec.min_value is None:
                errors.append(f"{prefix}: min_value required for {spec.parameter_type}")
            if spec.max_value is None:
                errors.append(f"{prefix}: max_value required for {spec.parameter_type}")
            if (spec.min_value is not None and 
                spec.max_value is not None and 
                spec.min_value >= spec.max_value):
                errors.append(f"{prefix}: min_value must be < max_value")
        
        elif spec.parameter_type == ParameterType.CATEGORICAL:
            if not spec.categorical_values:
                errors.append(f"{prefix}: categorical_values required for CATEGORICAL")
        
        elif spec.parameter_type == ParameterType.DISCRETE:
            if not spec.discrete_values:
                errors.append(f"{prefix}: discrete_values required for DISCRETE")
        
        return errors
    
    def generate_job_spec(self) -> Dict[str, Any]:
        """
        Generate the job specification that would be submitted.
        
        Returns:
            Job specification dictionary
        """
        worker_pool = self.config.get_worker_pool_spec()
        
        # Add dummy package URI for testing
        worker_pool["containerSpec"]["pythonPackageSpec"]["packageUris"] = [
            f"{self.config.staging_bucket}/packages/test/disgraphi-0.1.0.tar.gz"
        ]
        
        # Add GCS paths
        if self.config.data_dir_gcs:
            worker_pool["containerSpec"]["pythonPackageSpec"]["args"].extend([
                "--data_dir", self.config.data_dir_gcs
            ])
        
        if self.config.output_dir_gcs:
            worker_pool["containerSpec"]["pythonPackageSpec"]["args"].extend([
                "--output_dir", self.config.output_dir_gcs
            ])
        
        job_spec = {
            "displayName": self.config.display_name,
            "jobSpec": {
                "workerPoolSpecs": [worker_pool],
                "serviceAccount": self.config.service_account,
                "network": self.config.network,
                "timeout": f"{self.config.timeout_seconds}s",
            }
        }
        
        # Add hyperparameter tuning config if enabled
        hp_config = self.config.get_hyperparameter_tuning_config()
        if hp_config:
            job_spec["studySpec"] = hp_config["studySpec"]
            job_spec["maxTrialCount"] = hp_config["maxTrialCount"]
            job_spec["parallelTrialCount"] = hp_config["parallelTrialCount"]
        
        return job_spec
    
    def test_configuration(self, verbose: bool = True) -> bool:
        """
        Test the complete configuration.
        
        Args:
            verbose: Print detailed output
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if verbose:
            print("=" * 80)
            print("Testing Vertex AI Configuration")
            print("=" * 80)
        
        # Validate configuration
        errors = self.validate_config()
        
        if errors:
            if verbose:
                print("\n✗ Configuration validation FAILED:")
                for error in errors:
                    print(f"  - {error}")
            return False
        
        if verbose:
            print("\n✓ Configuration validation passed")
        
        # Generate job spec
        try:
            job_spec = self.generate_job_spec()
            
            if verbose:
                print("\n✓ Job specification generated successfully")
                print("\nJob Specification:")
                print("-" * 80)
                print(json.dumps(job_spec, indent=2))
                print("-" * 80)
            
            # Test training arguments
            args = self.config._build_training_args()
            if verbose:
                print("\nTraining Arguments:")
                print("-" * 80)
                print(" ".join(args))
                print("-" * 80)
            
            if self.config.enable_hyperparameter_tuning:
                if verbose:
                    print("\nHyperparameter Tuning Configuration:")
                    print("-" * 80)
                    print(f"Metric: {self.config.metric_id} ({self.config.metric_goal})")
                    print(f"Max trials: {self.config.max_trial_count}")
                    print(f"Parallel trials: {self.config.parallel_trial_count}")
                    print("\nParameters to tune:")
                    for spec in self.config.hyperparameter_specs:
                        print(f"  - {spec.parameter_id}:")
                        if spec.discrete_values:
                            print(f"    Values: {spec.discrete_values}")
                        elif spec.categorical_values:
                            print(f"    Values: {spec.categorical_values}")
                        else:
                            print(f"    Range: [{spec.min_value}, {spec.max_value}]")
                            print(f"    Scale: {spec.scale_type.value}")
                    print("-" * 80)
            
            if verbose:
                print("\n✓ All tests passed!")
                print("=" * 80)
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"\n✗ Error generating job specification: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def save_job_spec(self, output_path: str):
        """
        Save the job specification to a file.
        
        Args:
            output_path: Path to save the job spec JSON
        """
        job_spec = self.generate_job_spec()
        
        with open(output_path, 'w') as f:
            json.dump(job_spec, f, indent=2)
        
        print(f"Job specification saved to: {output_path}")
    
    def estimate_trial_combinations(self) -> int:
        """
        Estimate the total number of hyperparameter combinations.
        
        Returns:
            Number of possible combinations
        """
        if not self.config.enable_hyperparameter_tuning:
            return 1
        
        total = 1
        for spec in self.config.hyperparameter_specs:
            if spec.discrete_values:
                total *= len(spec.discrete_values)
            elif spec.categorical_values:
                total *= len(spec.categorical_values)
            else:
                # For continuous parameters, this is infinite
                # but we can estimate based on trial count
                total *= self.config.max_trial_count
        
        return min(total, self.config.max_trial_count)


def test_basic_config():
    """Test basic configuration without hyperparameter tuning."""
    config = VertexAIConfig(
        project_id="test-project",
        region="us-central1",
        base_args={
            'model_provider': 'smolvlm-256m',
            'num_train_epochs': 3,
            'learning_rate': 2e-5,
        }
    )
    
    tester = LocalTester(config)
    return tester.test_configuration()


def test_tuning_config():
    """Test configuration with hyperparameter tuning."""
    from .vertex_config import ParameterType
    
    config = VertexAIConfig(
        project_id="test-project",
        region="us-central1",
        enable_hyperparameter_tuning=True,
        hyperparameter_specs=[
            HyperparameterSpec(
                parameter_id="lora_r",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[4, 8, 16]
            ),
            HyperparameterSpec(
                parameter_id="learning_rate",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[1e-5, 2e-5, 5e-5]
            ),
        ],
        max_trial_count=9,
        parallel_trial_count=3,
        base_args={
            'model_provider': 'smolvlm-256m',
            'num_train_epochs': 3,
        }
    )
    
    tester = LocalTester(config)
    return tester.test_configuration()


if __name__ == '__main__':
    print("Running local tests...\n")
    
    print("Test 1: Basic configuration")
    if test_basic_config():
        print("✓ Passed\n")
    else:
        print("✗ Failed\n")
    
    print("\nTest 2: Hyperparameter tuning configuration")
    if test_tuning_config():
        print("✓ Passed\n")
    else:
        print("✗ Failed\n")
