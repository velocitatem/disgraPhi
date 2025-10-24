"""
SmolVLM Model Provider

SmolVLM is the smallest multimodal model in the world (256M-2.2B):
- Efficient vision-language model based on SmolLM2 + SigLIP
- Accepts image + text inputs to produce text outputs
- Designed for on-device applications with <1GB GPU RAM

Architecture:
- Vision Encoder: SigLIP (93M params for 256M model)
- Language Model: SmolLM2 (135M-1.7B)
- Visual Tokens: 64 tokens per 512×512 patch
- Training: LoRA on attention + MLP layers for efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

from .base import BaseVisionLanguageModel
from alveslib import get_logger

logger = get_logger(__name__)



# Model name mapping
MODEL_NAMES = {
    'smolvlm-256m': 'HuggingFaceTB/SmolVLM-256M-Instruct',
    'smolvlm-500m': 'HuggingFaceTB/SmolVLM-500M-Instruct',
    'smolvlm-2.2b': 'HuggingFaceTB/SmolVLM-Instruct',
}


class SmolVLMModel(BaseVisionLanguageModel):
    """
    SmolVLM model with LoRA adapter.

    Args:
        model_type: Model identifier (e.g., 'smolvlm-256m', 'smolvlm-500m')
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling factor (default: 16)
        lora_dropout: LoRA dropout (default: 0.05)
        load_in_4bit: Use 4-bit quantization (default: False, not needed for small models)
        load_in_8bit: Use 8-bit quantization (alternative to 4-bit)
        bootstrap_adapter_path: Path to frozen bootstrap adapter
        device_map: Device mapping strategy
        use_flash_attention: Use flash attention 2 if available (CUDA only)
    """

    def __init__(
        self,
        model_type: str = "smolvlm-256m",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bootstrap_adapter_path: Optional[str] = None,
        device_map: str = "auto",
        use_flash_attention: bool = True
    ):
        super().__init__()

        # Get HuggingFace model name
        self.model_name = MODEL_NAMES.get(model_type, MODEL_NAMES['smolvlm-256m'])
        self.model_type = model_type
        self.lora_config_dict = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        }

        # Load processor
        print(f"Loading processor from {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer

        # Configure quantization if enabled
        bnb_config = None
        if load_in_4bit or load_in_8bit:
            print(f"Configuring {'4-bit' if load_in_4bit else '8-bit'} quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None
            )

        # Determine attention implementation
        # attn_implementation = "flash_attention_2" if (
        #     use_flash_attention and torch.cuda.is_available()
        # ) else "eager"
        attn_implementation = "eager"  # Flash attention not supported for SmolVLM currently

        # Load base model
        print(f"Loading base model {self.model_name}...")
        print(f"Using attention implementation: {attn_implementation}")

        self.base_model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
            _attn_implementation=attn_implementation,
            device_map=device_map,
            quantization_config=bnb_config
        )

        # Prepare for k-bit training if quantized
        if load_in_4bit or load_in_8bit:
            print("Preparing model for k-bit training...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Enable gradient checkpointing to save memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

        # Configure LoRA
        # SmolVLM uses SmolLM2 as language model, target its attention layers
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
            task_type="CAUSAL_LM"
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
        self.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Image tensor [batch, channels, height, width]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for training [batch, seq_len]
            pixel_attention_mask: Pixel attention mask for images

        Returns:
            Dictionary with loss, logits, etc.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
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
        pixel_values: Union[torch.Tensor, Image.Image, str],
        prompt: str = "Transcribe this handwritten text:",
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate text from image + prompt.

        Args:
            pixel_values: Image tensor, PIL Image, or path to image
            prompt: Text prompt for the model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling (auto-set based on temperature)

        Returns:
            Generated text
        """
        logger.info("Preparing to process inputs")

        # Load image if path is provided
        if isinstance(pixel_values, str):
            from transformers.image_utils import load_image
            pixel_values = load_image(pixel_values)

        # Handle single image vs list of images
        images = [pixel_values] if not isinstance(pixel_values, list) else pixel_values

        # Create input messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt"
        )

        # Log input sizes
        if "pixel_values" in inputs:
            logger.info(f"Processed pixel_values shape: {inputs['pixel_values'].shape}")
        if "input_ids" in inputs:
            logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Move to device
        device = self.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Auto-determine sampling
        if do_sample is None:
            do_sample = temperature > 0

        # Generate
        logger.info("Starting generation")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample
            )

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("Generation complete")

        # Decode output
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        logger.info("Generated text successfully")

        # Extract response (SmolVLM uses chat format with Assistant: prefix)
        # The output will be in format "User: ... Assistant: <response>"
        if "Assistant:" in generated_text:
            transcription = generated_text.split("Assistant:")[-1].strip()
        elif prompt in generated_text:
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
        self.processor.save_pretrained(output_dir)
        print("✓ Adapter saved")

    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter from disk."""
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path
        )
        print("✓ Adapter loaded")

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        model_type: str = "smolvlm-256m",
        load_in_4bit: bool = False,
        device_map: str = "auto"
    ) -> 'SmolVLMModel':
        """
        Load model with pre-trained adapter.

        Args:
            adapter_path: Path to saved LoRA adapter
            model_type: Model type identifier
            load_in_4bit: Use 4-bit quantization
            device_map: Device mapping strategy

        Returns:
            Model instance with loaded adapter
        """
        # Load base model
        instance = cls(
            model_type=model_type,
            load_in_4bit=load_in_4bit,
            device_map=device_map
        )

        # Load adapter
        instance.load_adapter(adapter_path)

        return instance
