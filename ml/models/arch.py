"""
DisgraPhi Model Architecture

Qwen3-VL with LoRA for handwriting recognition:
- Bootstrap: Train on IAM database for general handwriting ability
- Personalization: Fine-tune per-user LoRA adapters

Architecture:
- Base: Qwen3-VL (vision-language model)
- Adapter: LoRA on attention + MLP layers
- Training: Two-stage (bootstrap frozen, personalization active)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

from alveslib import logger


class QwenVLHandwritingModel(nn.Module):
    """
    Qwen3-VL model with LoRA for handwriting recognition.

    Two-stage training:
    1. Bootstrap: Train base LoRA on IAM database
    2. Personalization: Train user-specific LoRA on top

    Args:
        model_name: Hugging Face model ID (default: Qwen/Qwen2-VL-7B-Instruct)
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling factor (default: 16)
        lora_dropout: LoRA dropout (default: 0.05)
        load_in_4bit: Use 4-bit quantization for memory efficiency
        bootstrap_adapter_path: Path to frozen bootstrap adapter (for personalization)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-4B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        bootstrap_adapter_path: Optional[str] = None,
        device_map: str = "auto"
    ):
        super().__init__()

        self.model_name = model_name
        self.lora_config = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        }

        # Load tokenizer and processor
        print(f"Loading tokenizer and processor from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Configure quantization if enabled
        if load_in_4bit:
            print("Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # Load base model
        print(f"Loading base model {model_name}...")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None
        )

        # Prepare for k-bit training if quantized
        if load_in_4bit:
            print("Preparing model for k-bit training...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Enable gradient checkpointing to save memory
        self.base_model.gradient_checkpointing_enable()

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                'q_proj',    # Query projection
                'k_proj',    # Key projection
                'v_proj',    # Value projection
                'o_proj',    # Output projection
                'gate_proj', # MLP gate
                'up_proj',   # MLP up
                'down_proj'  # MLP down
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"  # Qwen2-VL uses causal language modeling for text generation
        )

        # Apply LoRA
        print("Applying LoRA adapter...")
        self.model = get_peft_model(self.base_model, lora_config)

        # Load bootstrap adapter if provided (for personalization mode)
        if bootstrap_adapter_path:
            print(f"Loading bootstrap adapter from {bootstrap_adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                bootstrap_adapter_path,
                is_trainable=False  # Freeze bootstrap adapter
            )

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Image tensor [batch, channels, height, width]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for training [batch, seq_len]
            image_grid_thw: Image grid dimensions (temporal, height, width) for Qwen2-VL

        Returns:
            Dictionary with loss, logits, etc.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }

    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "Transcribe this handwritten text:",
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_pixels: Optional[int] = None
    ) -> str:
        """
        Generate transcription for a handwritten line image.

        Args:
            pixel_values: Image tensor [1, channels, height, width]
            prompt: Text prompt for the model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_pixels: Maximum number of pixels to process (limits memory usage)
                       Default: None (uses processor default, typically 12845056)
                       Reduce for large images to prevent OOM

        Returns:
            Transcribed text
        """
        # Prepare input with controlled resolution
        logger.info("Preparing to process inputs")

        # Build processor kwargs with memory limits
        processor_kwargs = {
            "text": [prompt],
            "images": pixel_values,
            "return_tensors": "pt"
        }

        # Add max_pixels if specified to limit vision token count
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
            logger.info(f"Limiting image to {max_pixels} pixels to control memory")

        inputs = self.processor(**processor_kwargs)

        # Log input size for debugging
        if "pixel_values" in inputs:
            pv_shape = inputs["pixel_values"].shape
            logger.info(f"Processed pixel_values shape: {pv_shape}")
        if "input_ids" in inputs:
            logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        logger.info("Inputs processed")

        # Move inputs to device one tensor at a time to avoid memory spike
        device = self.model.device
        inputs_on_device = {}

        for key, value in inputs.items():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inputs_on_device[key] = value.to(device)

        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate with controlled memory
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs_on_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("Generation complete")

        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        logger.info("Generated text successfully")

        # Extract transcription (remove prompt)
        if prompt in generated_text:
            transcription = generated_text.split(prompt)[-1].strip()
        else:
            transcription = generated_text.strip()

        return transcription

    def save_adapter(self, output_dir: str) -> None:
        """Save only the LoRA adapter weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving LoRA adapter to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("✓ Adapter saved")

    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter from disk."""
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path
        )
        print("✓ Adapter loaded")

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ) -> 'QwenVLHandwritingModel':
        """
        Load model with pre-trained adapter.

        Args:
            adapter_path: Path to saved LoRA adapter
            model_name: Base model name
            load_in_4bit: Use 4-bit quantization
            device_map: Device mapping strategy

        Returns:
            Model instance with loaded adapter
        """
        # Load base model
        instance = cls(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            device_map=device_map
        )

        # Load adapter
        instance.load_adapter(adapter_path)

        return instance


def create_bootstrap_model(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    load_in_4bit: bool = True
) -> QwenVLHandwritingModel:
    """
    Create model for bootstrap training on IAM database.

    Args:
        model_name: Hugging Face model ID
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        load_in_4bit: Use quantization

    Returns:
        Model instance ready for bootstrap training
    """
    print("=" * 60)
    print("Creating Bootstrap Model")
    print("=" * 60)

    model = QwenVLHandwritingModel(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
        bootstrap_adapter_path=None
    )

    return model


def create_personalization_model(
    bootstrap_adapter_path: str,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    load_in_4bit: bool = True
) -> QwenVLHandwritingModel:
    """
    Create model for personalization training on user data.

    The bootstrap adapter is frozen, and a new user-specific adapter is trained.

    Args:
        bootstrap_adapter_path: Path to frozen bootstrap adapter
        model_name: Hugging Face model ID
        lora_r: LoRA rank for user adapter
        lora_alpha: LoRA alpha for user adapter
        load_in_4bit: Use quantization

    Returns:
        Model instance ready for personalization training
    """
    print("=" * 60)
    print("Creating Personalization Model")
    print("=" * 60)

    model = QwenVLHandwritingModel(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
        bootstrap_adapter_path=bootstrap_adapter_path
    )

    return model


if __name__ == '__main__':
    """Test model loading and forward pass."""
    import argparse

    parser = argparse.ArgumentParser(description="Test DisgraPhi model architecture")
    parser.add_argument('--mode', choices=['bootstrap', 'personalization'], default='bootstrap')
    parser.add_argument('--bootstrap-path', type=str, help='Path to bootstrap adapter')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    args = parser.parse_args()

    if args.mode == 'bootstrap':
        model = create_bootstrap_model(model_name=args.model_name)
    else:
        if not args.bootstrap_path:
            raise ValueError("--bootstrap-path required for personalization mode")
        model = create_personalization_model(
            bootstrap_adapter_path=args.bootstrap_path,
            model_name=args.model_name
        )

    print("\n✓ Model loaded successfully")
    print(f"  Device: {next(model.model.parameters()).device}")
    print(f"  Dtype: {next(model.model.parameters()).dtype}")
