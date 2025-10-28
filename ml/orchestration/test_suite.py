#!/usr/bin/env python3
"""
Comprehensive test suite for the orchestration module.

Run this to validate all functionality before deploying to Vertex AI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.orchestration import (
    VertexAIConfig,
    HyperparameterSpec,
    ParameterType,
    ScaleType,
    VertexAILauncher
)
from ml.orchestration.local_test import LocalTester
from ml.orchestration.examples import (
    get_basic_training_config,
    get_bootstrap_tuning_config,
    get_personalization_config,
    get_multi_model_comparison_config,
    get_large_scale_config,
)


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...", end=" ")
    # Already imported above
    print("✓")
    return True


def test_basic_config():
    """Test basic configuration without hyperparameter tuning."""
    print("Testing basic configuration...", end=" ")
    
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
    errors = tester.validate_config()
    
    if errors:
        print(f"✗ Validation errors: {errors}")
        return False
    
    # Generate job spec
    spec = tester.generate_job_spec()
    assert 'displayName' in spec
    assert 'jobSpec' in spec
    assert 'workerPoolSpecs' in spec['jobSpec']
    
    print("✓")
    return True


def test_hyperparameter_tuning():
    """Test hyperparameter tuning configuration."""
    print("Testing hyperparameter tuning...", end=" ")
    
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
    errors = tester.validate_config()
    
    if errors:
        print(f"✗ Validation errors: {errors}")
        return False
    
    # Generate job spec
    spec = tester.generate_job_spec()
    assert 'studySpec' in spec
    assert 'maxTrialCount' in spec
    assert spec['maxTrialCount'] == 9
    assert spec['parallelTrialCount'] == 3
    
    print("✓")
    return True


def test_parameter_types():
    """Test different hyperparameter types."""
    print("Testing parameter types...", end=" ")
    
    # Test DISCRETE
    spec1 = HyperparameterSpec(
        parameter_id="lora_r",
        parameter_type=ParameterType.DISCRETE,
        discrete_values=[4, 8, 16]
    )
    assert spec1.to_vertex_dict()['parameterType'] == 'DISCRETE'
    
    # Test CATEGORICAL
    spec2 = HyperparameterSpec(
        parameter_id="model_provider",
        parameter_type=ParameterType.CATEGORICAL,
        categorical_values=["smolvlm-256m", "qwen3-vl-2b"]
    )
    assert spec2.to_vertex_dict()['parameterType'] == 'CATEGORICAL'
    
    # Test DOUBLE
    spec3 = HyperparameterSpec(
        parameter_id="learning_rate",
        parameter_type=ParameterType.DOUBLE,
        min_value=1e-6,
        max_value=1e-4,
        scale_type=ScaleType.UNIT_LOG_SCALE
    )
    assert spec3.to_vertex_dict()['parameterType'] == 'DOUBLE'
    
    # Test INTEGER
    spec4 = HyperparameterSpec(
        parameter_id="lora_r",
        parameter_type=ParameterType.INTEGER,
        min_value=4,
        max_value=32
    )
    assert spec4.to_vertex_dict()['parameterType'] == 'INTEGER'
    
    print("✓")
    return True


def test_validation_errors():
    """Test that validation catches errors."""
    print("Testing validation errors...", end=" ")
    
    # Missing project_id
    config = VertexAIConfig(
        project_id="",
        base_args={}
    )
    tester = LocalTester(config)
    errors = tester.validate_config()
    assert len(errors) > 0, "Should catch empty project_id"
    
    # Hyperparameter tuning without specs
    config2 = VertexAIConfig(
        project_id="test",
        enable_hyperparameter_tuning=True,
        hyperparameter_specs=[],
        base_args={}
    )
    tester2 = LocalTester(config2)
    errors2 = tester2.validate_config()
    assert len(errors2) > 0, "Should catch missing hyperparameter specs"
    
    # Invalid parallel trials
    config3 = VertexAIConfig(
        project_id="test",
        enable_hyperparameter_tuning=True,
        hyperparameter_specs=[
            HyperparameterSpec(
                parameter_id="lora_r",
                parameter_type=ParameterType.DISCRETE,
                discrete_values=[4, 8]
            )
        ],
        max_trial_count=2,
        parallel_trial_count=5,
        base_args={}
    )
    tester3 = LocalTester(config3)
    errors3 = tester3.validate_config()
    assert len(errors3) > 0, "Should catch parallel_trial_count > max_trial_count"
    
    print("✓")
    return True


def test_example_configs():
    """Test all example configurations."""
    print("Testing example configurations...")
    
    PROJECT_ID = 'test-project'
    
    configs = [
        ('  Basic Training', get_basic_training_config(PROJECT_ID)),
        ('  Bootstrap Tuning', get_bootstrap_tuning_config(PROJECT_ID)),
        ('  Personalization', get_personalization_config(
            PROJECT_ID,
            'gs://test/bootstrap',
            'gs://test/data'
        )),
        ('  Multi-Model Comparison', get_multi_model_comparison_config(PROJECT_ID)),
        ('  Large Scale', get_large_scale_config(PROJECT_ID)),
    ]
    
    all_passed = True
    for name, config in configs:
        print(f"{name}...", end=" ")
        tester = LocalTester(config)
        errors = tester.validate_config()
        if errors:
            print(f"✗ Errors: {errors}")
            all_passed = False
        else:
            print("✓")
    
    return all_passed


def test_job_spec_generation():
    """Test job specification generation."""
    print("Testing job spec generation...", end=" ")
    
    config = VertexAIConfig(
        project_id="test-project",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2,
        base_args={
            'model_provider': 'smolvlm-256m',
            'num_train_epochs': 5,
        }
    )
    
    tester = LocalTester(config)
    spec = tester.generate_job_spec()
    
    # Check machine spec
    worker_pool = spec['jobSpec']['workerPoolSpecs'][0]
    machine_spec = worker_pool['machineSpec']
    
    assert machine_spec['machineType'] == 'n1-standard-4'
    assert machine_spec['acceleratorType'] == 'NVIDIA_TESLA_V100'
    assert machine_spec['acceleratorCount'] == 2
    
    # Check training args
    args = worker_pool['containerSpec']['pythonPackageSpec']['args']
    assert '--model_provider' in args
    assert 'smolvlm-256m' in args
    assert '--num_train_epochs' in args
    assert '5' in args
    
    print("✓")
    return True


def test_training_args_building():
    """Test training arguments building."""
    print("Testing training args building...", end=" ")
    
    config = VertexAIConfig(
        project_id="test",
        base_args={
            'model_provider': 'smolvlm-256m',
            'num_train_epochs': 3,
            'learning_rate': 2e-5,
            'bf16': True,
            'fp16': False,
        }
    )
    
    args = config._build_training_args()
    
    # Check all args are present
    assert '--model_provider' in args
    assert 'smolvlm-256m' in args
    assert '--num_train_epochs' in args
    assert '3' in args
    assert '--learning_rate' in args
    assert '2e-05' in args
    assert '--bf16' in args
    # Boolean True should not have a value
    idx = args.index('--bf16')
    assert idx == len(args) - 1 or args[idx + 1].startswith('--')
    
    print("✓")
    return True


def test_package_creation():
    """Test that package can be created."""
    print("Testing package creation...", end=" ")
    
    config = VertexAIConfig(
        project_id="test-project",
        base_args={}
    )
    
    # This will fail if gcloud is not installed, which is expected
    # in local dev environments
    try:
        launcher = VertexAILauncher(config)
        # Try to package code (won't upload)
        package_path = launcher.package_training_code()
        assert Path(package_path).exists()
        assert package_path.endswith('.tar.gz')
        print("✓")
        return True
    except RuntimeError as e:
        if "gcloud" in str(e):
            print("⚠ (gcloud not installed, skipping)")
            return True
        raise


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("Running Orchestration Module Test Suite")
    print("=" * 80)
    print()
    
    tests = [
        test_imports,
        test_basic_config,
        test_hyperparameter_tuning,
        test_parameter_types,
        test_validation_errors,
        test_example_configs,
        test_job_spec_generation,
        test_training_args_building,
        test_package_creation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print()
    print("=" * 80)
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("=" * 80)
        return 0
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
