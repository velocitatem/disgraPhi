"""
Florence-2 Model Provider

Florence-2 is a vision foundation model using prompt-based approach for various vision tasks:
- Image captioning (caption, detailed caption, more detailed caption)
- Object detection and region proposal
- OCR and OCR with region
- Caption to phrase grounding
- Dense region captioning

Architecture:
- Unified sequence-to-sequence architecture
- Trained on FLD-5B dataset (5.4B annotations across 126M images)
- Supports both zero-shot and fine-tuned settings
- Base: 0.23B params, Large: 0.77B params

Key Features:
- Multi-task learning via text prompts
- Competitive zero-shot performance
- Efficient fine-tuning with LoRA
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
from PIL import Image
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


# Model name mapping
MODEL_NAMES = {
    'florence2-base': 'microsoft/Florence-2-base',
    'florence2-large': 'microsoft/Florence-2-large',
    'florence2-base-ft': 'microsoft/Florence-2-base-ft',
    'florence2-large-ft': 'microsoft/Florence-2-large-ft',
}

# Task prompt mapping
TASK_PROMPTS = {
    'caption': '<CAPTION>',
    'detailed_caption': '<DETAILED_CAPTION>',
    'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
    'od': '<OD>',  # Object Detection
    'dense_region_caption': '<DENSE_REGION_CAPTION>',
    'region_proposal': '<REGION_PROPOSAL>',
    'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
    'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
    'region_to_segmentation': '<REGION_TO_SEGMENTATION>',
    'open_vocabulary_detection': '<OPEN_VOCABULARY_DETECTION>',
    'region_to_category': '<REGION_TO_CATEGORY>',
    'region_to_description': '<REGION_TO_DESCRIPTION>',
    'ocr': '<OCR>',
    'ocr_with_region': '<OCR_WITH_REGION>',
}


class Florence2Model(BaseVisionLanguageModel):
    """
    Florence-2 vision foundation model with LoRA adapter.

    Args:
        model_type: Model identifier (e.g., 'florence2-base', 'florence2-large')
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling factor (default: 16)
        lora_dropout: LoRA dropout (default: 0.05)
        load_in_4bit: Use 4-bit quantization (default: False)
        load_in_8bit: Use 8-bit quantization (alternative to 4-bit)
        bootstrap_adapter_path: Path to frozen bootstrap adapter
        device_map: Device mapping strategy
        trust_remote_code: Required for Florence-2 (default: True)
    """

    def __init__(
        self,
        model_type: str = "florence2-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bootstrap_adapter_path: Optional[str] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True
    ):
        super().__init__()

        # Get HuggingFace model name
        self.model_name = MODEL_NAMES.get(model_type, MODEL_NAMES['florence2-base'])
        self.model_type = model_type
        self.lora_config_dict = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        }

        # Load processor
        print(f"Loading processor from {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code
        )
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

        # Configure quantization if enabled
        bnb_config = None
        if load_in_4bit or load_in_8bit:
            print(f"Configuring {'4-bit' if load_in_4bit else '8-bit'} quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
            )

        # Determine dtype
        model_dtype = torch.float16 if not (load_in_4bit or load_in_8bit) else None

        # Load base model
        print(f"Loading base model {self.model_name}...")
        # Florence-2 has some compatibility issues with attention implementations
        # We need to explicitly set attn_implementation or handle the _supports_sdpa issue
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            attn_implementation="eager",  # Use eager attention to avoid SDPA compatibility issues
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=trust_remote_code
        )

        # Prepare for k-bit training if quantized
        if load_in_4bit or load_in_8bit:
            print("Preparing model for k-bit training...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Enable gradient checkpointing to save memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

        # Configure LoRA
        # Florence-2 uses encoder-decoder architecture, target attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                'q_proj',    # Query projection
                'k_proj',    # Key projection
                'v_proj',    # Value projection
                'out_proj',  # Output projection
                'fc1',       # Feed-forward layer 1
                'fc2',       # Feed-forward layer 2
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
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Image tensor [batch, channels, height, width]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for training [batch, seq_len]

        Returns:
            Dictionary with loss, logits, etc.
        """
        # Ensure pixel_values match model dtype
        # Florence-2 requires inputs to be in the same dtype as model weights
        if pixel_values is not None and pixel_values.dtype != self.dtype:
            pixel_values = pixel_values.to(self.dtype)

        outputs = self.model(
            pixel_values=pixel_values,
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
        prompt: str = "<OCR>",
        text_input: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        num_beams: int = 3,
        do_sample: bool = False,
        return_scores: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Generate text/predictions from image + prompt.

        Args:
            pixel_values: Image tensor, PIL Image, or path to image
            prompt: Task prompt (e.g., '<CAPTION>', '<OD>', '<OCR>')
            text_input: Additional text input for tasks like caption-to-phrase grounding
            max_new_tokens: Maximum tokens to generate (default: 1024 for Florence-2)
            temperature: Sampling temperature
            num_beams: Number of beams for beam search (default: 3)
            do_sample: Whether to use sampling
            return_scores: Whether to return confidence scores

        Returns:
            Generated text or (text, parsed_result) if applicable
        """
        text_input = ""
        logger.info(f"Running Florence-2 inference with task: {prompt}")

        # Load image if path is provided
        if isinstance(pixel_values, str):
            pixel_values = Image.open(pixel_values).convert('RGB')

        # Build complete prompt
        if text_input is not None:
            full_prompt = prompt + text_input
        else:
            full_prompt = prompt

        # Process inputs
        inputs = self.processor(
            text=full_prompt,
            images=pixel_values,
            return_tensors="pt"
        )

        # Log input sizes
        if "pixel_values" in inputs:
            logger.info(f"Processed pixel_values shape: {inputs['pixel_values'].shape}")
        if "input_ids" in inputs:
            logger.info(f"Input sequence length: {inputs['input_ids'].shape[1]}")

        # Move to device and ensure correct dtype
        device = self.device
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

        # Ensure pixel_values match model dtype (fix float32 vs float16 mismatch)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            logger.info(f"Converted pixel_values to {self.dtype}")

        # Generate
        logger.info("Starting generation")
        logger.info(f"Generate params: max_new_tokens={max_new_tokens}, num_beams={num_beams}, do_sample={do_sample}")

        # Temporarily disable gradient checkpointing for generation if enabled
        # Gradient checkpointing + use_cache causes issues
        is_grad_checkpointing = getattr(self.base_model.config, 'use_cache', None) is False
        if is_grad_checkpointing:
            logger.info("Temporarily disabling gradient checkpointing for generation")

        try:
            with torch.no_grad():
                # Florence-2 with LoRA has generation issues
                # Try without use_cache first (gradient checkpointing incompatibility)
                gen_kwargs = {
                    **inputs,
                    'max_new_tokens': max_new_tokens,
                    'num_beams': num_beams,
                    'do_sample': do_sample,
                }

                # First attempt: try with use_cache=False (safer with LoRA)
                try:
                    logger.info("Attempting generation with use_cache=False")
                    if return_scores:
                        generated_output = self.model.generate(
                            **gen_kwargs,
                            use_cache=False,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        generated_ids = generated_output.sequences
                    else:
                        generated_ids = self.model.generate(
                            **gen_kwargs,
                            use_cache=False
                        )
                except Exception as e1:
                    logger.warning(f"Generation with use_cache=False failed: {e1}")
                    logger.info("Retrying with use_cache=True")
                    # Second attempt: try with use_cache=True (default)
                    if return_scores:
                        generated_output = self.model.generate(
                            **gen_kwargs,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        generated_ids = generated_output.sequences
                    else:
                        generated_ids = self.model.generate(
                            **gen_kwargs,
                            use_cache=True
                        )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.error(f"Input IDs shape: {inputs['input_ids'].shape}")
            logger.error(f"Pixel values shape: {inputs['pixel_values'].shape if inputs.get('pixel_values') is not None else 'None'}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        if generated_ids is None:
            logger.error("Generation returned None!")
            raise ValueError("Model generation returned None")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info(f"Generation complete - generated {generated_ids.shape}")

        # Decode output
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        logger.info("Decoded text successfully")

        # Post-process generation if it's a structured task
        if isinstance(pixel_values, Image.Image):
            image_size = (pixel_values.width, pixel_values.height)
        else:
            # Assume standard size or extract from tensor
            image_size = (1000, 1000)  # Default fallback

        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=image_size
            )

            # If return_scores is requested and we have scores
            if return_scores and hasattr(generated_output, 'scores'):
                # Compute transition scores for confidence
                transition_beam_scores = self.model.compute_transition_scores(
                    sequences=generated_output.sequences,
                    scores=generated_output.scores,
                    beam_indices=generated_output.beam_indices,
                )
                parsed_answer['confidence_scores'] = transition_beam_scores[0].cpu().tolist()

            # Extract clean text from parsed answer if available
            # For OCR tasks, the parsed answer contains the clean text
            if isinstance(parsed_answer, dict):
                # Try to get the text from the parsed answer
                if prompt in parsed_answer:
                    clean_text = parsed_answer[prompt]
                elif 'text' in parsed_answer:
                    clean_text = parsed_answer['text']
                else:
                    # Fallback to raw generated text
                    clean_text = generated_text
            else:
                clean_text = generated_text

            return clean_text, parsed_answer
        except Exception as e:
            logger.warning(f"Could not post-process generation: {e}")
            return generated_text

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

    def run_task(
        self,
        image: Union[Image.Image, str],
        task: str,
        text_input: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method to run specific Florence-2 tasks.

        Args:
            image: PIL Image or path to image
            task: Task name (e.g., 'caption', 'od', 'ocr')
            text_input: Additional text input for certain tasks
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with parsed results
        """
        # Get task prompt
        prompt = TASK_PROMPTS.get(task, task)

        # Run generation
        result = self.generate(
            pixel_values=image,
            prompt=prompt,
            text_input=text_input,
            **kwargs
        )

        # Return parsed result if available
        if isinstance(result, tuple):
            return result[1]
        else:
            return {'text': result}

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        model_type: str = "florence2-base",
        load_in_4bit: bool = False,
        device_map: str = "auto"
    ) -> 'Florence2Model':
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
