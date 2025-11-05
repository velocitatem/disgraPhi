"""
Unified configuration system for DisgraPhi.

Supports:
- Loading and merging YAML configs
- VRAM budget calculation
- Automatic optimization recommendations
- CLI override support
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import warnings


@dataclass
class VRAMBudget:
    """VRAM budget breakdown."""
    model_gb: float
    lora_gb: float
    optimizer_gb: float
    activations_gb: float
    gradients_gb: float
    buffer_gb: float = 1.0

    @property
    def total_gb(self) -> float:
        return (self.model_gb + self.lora_gb + self.optimizer_gb +
                self.activations_gb + self.gradients_gb + self.buffer_gb)

    def fits_in_vram(self, available_gb: float) -> bool:
        return self.total_gb <= available_gb

    def __str__(self) -> str:
        return f"""VRAM Budget Breakdown:
  Model:        {self.model_gb:.2f} GB
  LoRA adapters:{self.lora_gb:.2f} GB
  Optimizer:    {self.optimizer_gb:.2f} GB
  Activations:  {self.activations_gb:.2f} GB
  Gradients:    {self.gradients_gb:.2f} GB
  Buffer:       {self.buffer_gb:.2f} GB
  {"="*40}
  Total:        {self.total_gb:.2f} GB
"""


def calculate_vram_budget(
    model_size_b: float,
    quantization: str,
    lora_r: int,
    batch_size: int,
    use_liger_kernel: bool = False,
    gradient_checkpointing: bool = True
) -> VRAMBudget:
    """
    Calculate estimated VRAM usage.

    Args:
        model_size_b: Model size in billions of parameters
        quantization: '4bit', '8bit', 'bf16', 'fp32', or 'none'
        lora_r: LoRA rank
        batch_size: Training batch size
        use_liger_kernel: Whether Liger Kernel optimizations are enabled
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        VRAMBudget object with detailed breakdown
    """
    # Quantization multipliers (bytes per parameter)
    quant_multipliers = {
        '4bit': 0.5,
        '8bit': 1.0,
        'bf16': 2.0,
        'fp16': 2.0,
        'fp32': 4.0,
        'none': 2.0  # Default to bf16
    }

    bytes_per_param = quant_multipliers.get(quantization, 2.0)
    model_gb = model_size_b * bytes_per_param

    # LoRA parameters (rough estimate: r * model_size * 0.0001)
    lora_params_b = lora_r * model_size_b * 0.0001
    lora_gb = lora_params_b * 2.0  # LoRA in fp16/bf16

    # Optimizer states (depends on optimizer type)
    # For AdamW8bit: ~2x LoRA params (vs 8x for standard AdamW)
    optimizer_gb = lora_gb * 2.0  # Assuming 8-bit optimizer

    # Activations (depends on batch size and model depth)
    # Rough heuristic: model_size * 0.5 * batch_size
    # Reduced by gradient checkpointing (~70% reduction)
    activation_multiplier = 0.15 if gradient_checkpointing else 0.5
    activations_gb = model_size_b * activation_multiplier * batch_size

    # Gradients (for LoRA parameters only)
    gradients_gb = lora_gb * batch_size * 0.5

    # Apply Liger Kernel reduction (60% overall)
    if use_liger_kernel:
        activations_gb *= 0.4
        gradients_gb *= 0.4

    return VRAMBudget(
        model_gb=model_gb,
        lora_gb=lora_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        gradients_gb=gradients_gb
    )


def recommend_batch_size(
    model_size_b: float,
    quantization: str,
    lora_r: int,
    available_vram_gb: float,
    use_liger_kernel: bool = False,
    gradient_checkpointing: bool = True
) -> int:
    """
    Recommend optimal batch size for given VRAM.

    Returns:
        Recommended batch size (1-8)
    """
    for batch_size in [8, 6, 4, 3, 2, 1]:
        budget = calculate_vram_budget(
            model_size_b, quantization, lora_r, batch_size,
            use_liger_kernel, gradient_checkpointing
        )
        if budget.fits_in_vram(available_vram_gb):
            return batch_size
    return 1


class ConfigLoader:
    """Load and merge configuration files."""

    def __init__(self, config_dir: Path = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = Path(config_dir)

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a single YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple config dictionaries.
        Later configs override earlier ones.
        """
        result = {}
        for config in configs:
            self._deep_merge(result, config)
        return result

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load a preset configuration.

        Args:
            preset_name: Name of preset (e.g., 'laptop_bootstrap_deepseek')
                        or path to preset YAML file

        Returns:
            Merged configuration dictionary
        """
        if preset_name.endswith('.yaml'):
            preset_path = Path(preset_name)
        else:
            preset_path = self.config_dir / "presets" / f"{preset_name}.yaml"

        return self.load_yaml(preset_path)

    def load_composed(
        self,
        hardware: Optional[str] = None,
        model: Optional[str] = None,
        training: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load and merge individual config components.

        Args:
            hardware: Hardware config name (e.g., 'laptop_12gb')
            model: Model config name (e.g., 'deepseek_ocr')
            training: Training config name (e.g., 'bootstrap')

        Returns:
            Merged configuration dictionary
        """
        configs = []

        if hardware:
            hw_path = self.config_dir / "hardware" / f"{hardware}.yaml"
            configs.append(self.load_yaml(hw_path))

        if model:
            model_path = self.config_dir / "models" / f"{model}.yaml"
            configs.append(self.load_yaml(model_path))

        if training:
            train_path = self.config_dir / "training" / f"{training}.yaml"
            configs.append(self.load_yaml(train_path))

        return self.merge_configs(*configs)

    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CLI overrides to config.

        Args:
            config: Base configuration
            overrides: Dictionary of override values (flat or nested)

        Returns:
            Updated configuration
        """
        result = config.copy()
        for key, value in overrides.items():
            # Support dot notation: "training.learning_rate" -> {"training": {"learning_rate": ...}}
            if '.' in key:
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                result[key] = value
        return result

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []

        # Check required fields
        if 'model' not in config or 'provider' not in config.get('model', {}):
            issues.append("ERROR: model.provider is required")

        if 'training' not in config:
            issues.append("ERROR: training configuration is required")

        # Check VRAM budget
        if 'hardware' in config and 'vram_gb' in config['hardware']:
            available_vram = config['hardware']['vram_gb']
            model_config = config.get('model', {})
            training_config = config.get('training', {})

            # Estimate model size from provider
            model_provider = model_config.get('provider', '')
            model_size_b = self._estimate_model_size(model_provider)

            if model_size_b > 0:
                quantization = model_config.get('quantization', 'none')
                lora_r = model_config.get('lora', {}).get('r', 16)
                batch_size = training_config.get('per_device_train_batch_size', 2)
                use_liger = config.get('memory', {}).get('use_liger_kernel', False)
                grad_checkpoint = config.get('memory', {}).get('gradient_checkpointing', True)

                budget = calculate_vram_budget(
                    model_size_b, quantization, lora_r, batch_size,
                    use_liger, grad_checkpoint
                )

                if not budget.fits_in_vram(available_vram):
                    issues.append(f"WARNING: Estimated VRAM usage ({budget.total_gb:.1f}GB) exceeds available ({available_vram}GB)")
                    recommended = recommend_batch_size(
                        model_size_b, quantization, lora_r, available_vram,
                        use_liger, grad_checkpoint
                    )
                    issues.append(f"  Recommended: Reduce batch_size to {recommended}")

        return issues

    def _estimate_model_size(self, provider: str) -> float:
        """Estimate model size in billions of parameters from provider name."""
        size_map = {
            'deepseek-ocr': 3.0,
            'smolvlm-256m': 0.256,
            'smolvlm-500m': 0.5,
            'smolvlm-2.2b': 2.2,
            'qwen3-vl-2b': 2.0,
            'qwen3-vl-4b': 4.0,
            'qwen3-vl-7b': 7.0,
        }
        return size_map.get(provider, 0.0)

    def print_config_summary(self, config: Dict[str, Any]) -> None:
        """Print human-readable config summary."""
        print("=" * 80)
        print("Configuration Summary")
        print("=" * 80)

        # Model
        model = config.get('model', {})
        print(f"Model: {model.get('provider', 'unknown')}")
        print(f"  Quantization: {model.get('quantization', 'none')}")
        lora = model.get('lora', {})
        print(f"  LoRA: r={lora.get('r', 8)}, alpha={lora.get('alpha', 16)}")

        # Training
        training = config.get('training', {})
        print(f"\nTraining: {training.get('mode', 'unknown')}")
        print(f"  Dataset: {training.get('dataset', 'unknown')}")
        print(f"  Epochs: {training.get('num_train_epochs', 'unknown')}")
        print(f"  Batch size: {training.get('per_device_train_batch_size', 'unknown')}")
        print(f"  Learning rate: {training.get('learning_rate', 'unknown')}")

        # Hardware
        hardware = config.get('hardware', {})
        if hardware:
            print(f"\nHardware:")
            print(f"  VRAM: {hardware.get('vram_gb', 'unknown')}GB")
            print(f"  Optimization: {hardware.get('optimization', 'unknown')}")

        # Memory
        memory = config.get('memory', {})
        if memory:
            print(f"\nMemory Optimizations:")
            print(f"  Liger Kernel: {memory.get('use_liger_kernel', False)}")
            print(f"  Gradient Checkpointing: {memory.get('gradient_checkpointing', True)}")
            print(f"  Optimizer: {memory.get('optim', 'adamw_torch')}")

        # VRAM Budget
        if hardware.get('vram_gb'):
            model_provider = model.get('provider', '')
            model_size_b = self._estimate_model_size(model_provider)
            if model_size_b > 0:
                budget = calculate_vram_budget(
                    model_size_b,
                    model.get('quantization', 'none'),
                    lora.get('r', 16),
                    training.get('per_device_train_batch_size', 2),
                    memory.get('use_liger_kernel', False),
                    memory.get('gradient_checkpointing', True)
                )
                print(f"\n{budget}")
                fits = "✓ YES" if budget.fits_in_vram(hardware['vram_gb']) else "✗ NO"
                print(f"Fits in {hardware['vram_gb']}GB VRAM: {fits}")

        print("=" * 80)


def load_config(
    preset: Optional[str] = None,
    hardware: Optional[str] = None,
    model: Optional[str] = None,
    training: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        preset: Preset config name or path
        hardware: Hardware config name
        model: Model config name
        training: Training config name
        overrides: CLI overrides

    Returns:
        Merged and validated configuration
    """
    loader = ConfigLoader()

    if preset:
        config = loader.load_preset(preset)
    else:
        config = loader.load_composed(hardware, model, training)

    if overrides:
        config = loader.apply_overrides(config, overrides)

    # Validate
    issues = loader.validate_config(config)
    for issue in issues:
        if issue.startswith("ERROR"):
            raise ValueError(issue)
        else:
            warnings.warn(issue)

    return config


if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader()

    # Test preset loading
    print("Testing preset loading...")
    config = loader.load_preset("laptop_bootstrap_deepseek")
    loader.print_config_summary(config)

    print("\n" + "=" * 80 + "\n")

    # Test composed loading
    print("Testing composed loading...")
    config = loader.load_composed(
        hardware="laptop_12gb",
        model="deepseek_ocr",
        training="bootstrap"
    )
    loader.print_config_summary(config)

    print("\n" + "=" * 80 + "\n")

    # Test VRAM calculation
    print("Testing VRAM budget calculator...")
    for batch_size in [1, 2, 4]:
        budget = calculate_vram_budget(
            model_size_b=3.0,
            quantization='4bit',
            lora_r=16,
            batch_size=batch_size,
            use_liger_kernel=True,
            gradient_checkpointing=True
        )
        print(f"Batch size {batch_size}: {budget.total_gb:.2f}GB")
