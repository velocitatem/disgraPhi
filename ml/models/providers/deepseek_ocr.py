"""
DeepSeek-OCR Model Provider

DeepSeek-OCR is a 3B parameter vision-language model optimized for OCR:
- Purpose-built for optical character recognition
- Document-to-markdown conversion
- Contexts optical compression for efficient visual-text processing
- Multiple size configurations (Tiny, Base, Large)

Architecture:
- 3B parameters (BF16)
- Vision encoder + Language decoder
- Optimized for handwriting recognition with LoRA fine-tuning
- Recommended: 50-200 samples for personalization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from PIL import Image
import os
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
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
logger.setLevel(os.getenv("DISGRAPHI_LOGLEVEL", "ERROR").upper())


# Model name mapping
MODEL_NAMES = {
    'deepseek-ocr': 'deepseek-ai/DeepSeek-OCR',
    'deepseek-ocr-tiny': 'deepseek-ai/DeepSeek-OCR-Tiny',
    'deepseek-ocr-base': 'deepseek-ai/DeepSeek-OCR',
    'deepseek-ocr-large': 'deepseek-ai/DeepSeek-OCR-Large',
}


class DeepSeekOCRModel(BaseVisionLanguageModel):
    """
    DeepSeek-OCR model with LoRA adapter.

    Args:
        model_type: Model identifier (e.g., 'deepseek-ocr', 'deepseek-ocr-large')
        lora_r: LoRA rank (default: 16, recommended 16-32 for handwriting)
        lora_alpha: LoRA scaling factor (default: 32, recommended 2×rank)
        lora_dropout: LoRA dropout (default: 0.05)
        load_in_4bit: Use 4-bit quantization (recommended for 12GB GPU)
        load_in_8bit: Use 8-bit quantization (alternative to 4-bit)
        bootstrap_adapter_path: Path to frozen bootstrap adapter
        device_map: Device mapping strategy
        finetune_vision_layers: Whether to fine-tune vision encoder (default: False)
        finetune_language_layers: Whether to fine-tune language decoder (default: True)
        attn_implementation: Attention implementation ("eager", "sdpa", or "flash_attention_2")
    """

    def __init__(
        self,
        model_type: str = "deepseek-ocr",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        bootstrap_adapter_path: Optional[str] = None,
        device_map: str = "auto",
        finetune_vision_layers: bool = False,
        finetune_language_layers: bool = True,
        attn_implementation: str = "eager"
    ):
        super().__init__()

        # Get HuggingFace model name
        self.model_name = MODEL_NAMES.get(model_type, MODEL_NAMES['deepseek-ocr'])
        self.model_type = model_type
        self.lora_config_dict = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        }
        self.finetune_vision_layers = finetune_vision_layers
        self.finetune_language_layers = finetune_language_layers

        # Load processor
        print(f"Loading processor from {self.model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load processor: {e}")
            print("Attempting to load tokenizer only...")
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

        self.tokenizer = self.processor if hasattr(self.processor, 'encode') else self.processor.tokenizer

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

        # Load base model
        print(f"Loading base model {self.model_name}...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )

        # Prepare for k-bit training if quantized
        if load_in_4bit or load_in_8bit:
            print("Preparing model for k-bit training...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Enable gradient checkpointing to save memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            if hasattr(self.base_model.config, 'use_cache'):
                self.base_model.config.use_cache = False

        # Configure LoRA target modules
        target_modules = []

        if finetune_language_layers:
            # Language decoder modules (standard attention + MLP)
            target_modules.extend([
                'q_proj',    # Query projection
                'k_proj',    # Key projection
                'v_proj',    # Value projection
                'o_proj',    # Output projection
                'gate_proj', # MLP gate
                'up_proj',   # MLP up
                'down_proj'  # MLP down
            ])

        if finetune_vision_layers:
            # Vision encoder modules (if accessible)
            # DeepSeek-OCR specific vision modules might differ
            target_modules.extend([
                'vision.q_proj',
                'vision.k_proj',
                'vision.v_proj',
                'vision.o_proj'
            ])

        if not target_modules:
            raise ValueError("Must enable either finetune_vision_layers or finetune_language_layers")

        print(f"LoRA configuration: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Fine-tuning: vision={finetune_vision_layers}, language={finetune_language_layers}")
        print(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Load bootstrap adapter if provided (for personalization mode)
        if bootstrap_adapter_path:
            print(f"Loading bootstrap adapter from {bootstrap_adapter_path}...")

            # Load bootstrap as frozen adapter
            print("Loading bootstrap adapter as frozen base...")
            bootstrap_model = PeftModel.from_pretrained(
                self.base_model,
                bootstrap_adapter_path,
                adapter_name="bootstrap",
                is_trainable=False
            )

            # Add trainable personalization adapter on top
            print("Adding trainable personalization adapter...")
            self.model = bootstrap_model
            self.model.add_adapter("personalization", lora_config)
            self.model.set_adapter("personalization")

            print("✓ Architecture: base_model + frozen_bootstrap + trainable_personalization")

            # Ensure bootstrap is frozen, personalization is trainable
            for name, param in self.model.named_parameters():
                if "bootstrap" in name:
                    param.requires_grad = False
                elif "personalization" in name or "lora" in name.lower():
                    param.requires_grad = True
        else:
            # Bootstrap training: just apply LoRA
            print("Applying LoRA adapter...")
            self.model = get_peft_model(self.base_model, lora_config)

        # Print trainable parameters
        self.print_trainable_parameters()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Image tensor (may be embedded in input_ids for DeepSeek-OCR)
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for training [batch, seq_len]

        Returns:
            Dictionary with loss, logits, etc.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
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
            Generated text (transcribed handwriting)
        """
        logger.info("Preparing to process inputs")

        # Load image if path is provided
        if isinstance(pixel_values, str):
            pixel_values = Image.open(pixel_values).convert('RGB')

        # Handle single image
        if isinstance(pixel_values, Image.Image):
            image = pixel_values
        else:
            # Assume it's already a tensor
            image = pixel_values

        # DeepSeek-OCR format: "<image>\nPrompt text"
        # The image is processed specially by the processor
        formatted_prompt = f"<image>\n{prompt}"

        # Process inputs
        if hasattr(self.processor, 'process_images'):
            # Custom processor with image handling
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            )
        else:
            # Fallback: tokenize text only
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True
            )

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
                do_sample=do_sample,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("Generation complete")

        # Decode output
        generated_text = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        logger.info("Generated text successfully")

        # Extract transcription (remove prompt)
        if prompt in generated_text:
            transcription = generated_text.split(prompt)[-1].strip()
        elif "<image>" in generated_text:
            transcription = generated_text.split("<image>")[-1].strip()
            if transcription.startswith("\n"):
                transcription = transcription[1:].strip()
        else:
            transcription = generated_text.strip()

        return transcription

    def save_adapter(self, output_dir: str) -> None:
        """Save only the LoRA adapter weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving LoRA adapter to {output_dir}...")
        self.model.save_pretrained(output_dir)
        if hasattr(self.processor, 'save_pretrained'):
            self.processor.save_pretrained(output_dir)
        print("✓ Adapter saved")

    def load_adapter(
        self,
        adapter_path: str,
        adapter_name: str = "default",
        is_trainable: bool = True
    ) -> None:
        """
        Load a LoRA adapter from disk.

        Args:
            adapter_path: Path to adapter weights
            adapter_name: Name for the adapter
            is_trainable: Whether adapter should be trainable
        """
        adapter_path = Path(adapter_path)

        print(f"Loading LoRA adapter '{adapter_name}' from {adapter_path}...")
        self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)

        # Set trainability
        if not is_trainable:
            for name, param in self.model.named_parameters():
                if adapter_name in name:
                    param.requires_grad = False
            print(f"✓ Adapter '{adapter_name}' loaded and frozen")
        else:
            print(f"✓ Adapter '{adapter_name}' loaded (trainable)")

    def set_adapter(self, adapter_names: List[str]) -> None:
        """
        Set active adapters for inference/training.

        Args:
            adapter_names: List of adapter names to activate
        """
        self.model.set_adapter(adapter_names)
        print(f"✓ Active adapters: {adapter_names}")

    def get_adapter_state_dict(self, adapter_name: str = "default") -> Dict[str, Any]:
        """
        Get state dict for a specific adapter.

        Args:
            adapter_name: Name of the adapter

        Returns:
            Adapter state dict
        """
        state_dict = {}
        for name, param in self.model.named_parameters():
            if adapter_name in name:
                state_dict[name] = param
        return state_dict

    def prepare_training_batch(self, images: List[Any], texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training with DeepSeek-OCR format.

        Args:
            images: List of PIL Images
            texts: List of ground truth texts

        Returns:
            Processed batch ready for forward pass
        """
        # DeepSeek-OCR training format: "<image>\nTranscribe: " + ground_truth
        formatted_inputs = []

        for text in texts:
            formatted_inputs.append(f"<image>\nTranscribe: {text}")

        # Process through processor
        if hasattr(self.processor, 'process_images'):
            inputs = self.processor(
                text=formatted_inputs,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
        else:
            # Fallback: text only
            inputs = self.tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )

        # Create labels (for causal LM, labels = input_ids shifted)
        labels = inputs['input_ids'].clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Optionally: mask the prompt part (only train on transcription)
        # For now, train on full sequence

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        model_type: str = "deepseek-ocr",
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ) -> 'DeepSeekOCRModel':
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
