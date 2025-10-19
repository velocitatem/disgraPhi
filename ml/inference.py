"""
DisgraPhi Inference API

FastAPI server for handwriting recognition inference using trained LoRA adapters.

Environment Variables:
    ML_ADAPTER_PATH: Path to LoRA adapter checkpoint (required)
    ML_MODEL_NAME: Hugging Face model name (default: Qwen/Qwen2-VL-7B-Instruct)
    ML_DEVICE: Device to run inference on (default: auto)
"""

import os
import io
import base64
from typing import Optional, List
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from alveslib import get_logger

from ml.models.arch import QwenVLHandwritingModel


logger = get_logger("ml-inference")


# ============================================================
# Configuration
# ============================================================

ADAPTER_PATH = os.getenv("ML_ADAPTER_PATH")
if ADAPTER_PATH is None:
    raise RuntimeError(
        "ML_ADAPTER_PATH environment variable not set. "
        "Point it to your LoRA adapter checkpoint directory."
    )

MODEL_NAME = os.getenv("ML_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
DEVICE = os.getenv("ML_DEVICE", "auto")

logger.info(f"Loading model from: {MODEL_NAME}")
logger.info(f"Loading adapter from: {ADAPTER_PATH}")
logger.info(f"Device: {DEVICE}")


# ============================================================
# Pydantic Models
# ============================================================

class ImageInput(BaseModel):
    """Input schema for single image transcription."""

    image: str = Field(
        ...,
        description="Base64-encoded image (PNG, JPG, or other PIL-supported format)"
    )
    max_length: Optional[int] = Field(
        128,
        description="Maximum number of tokens to generate",
        ge=1,
        le=512
    )
    temperature: Optional[float] = Field(
        0.0,
        description="Sampling temperature (0 = greedy decoding)",
        ge=0.0,
        le=2.0
    )


class BatchImageInput(BaseModel):
    """Input schema for batch image transcription."""

    images: List[str] = Field(
        ...,
        description="List of base64-encoded images"
    )
    max_length: Optional[int] = Field(128, ge=1, le=512)
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0)


class TranscriptionOutput(BaseModel):
    """Output schema for transcription result."""

    text: str = Field(..., description="Transcribed text from the image")
    confidence: Optional[float] = Field(
        None,
        description="Model confidence score (if available)"
    )


class BatchTranscriptionOutput(BaseModel):
    """Output schema for batch transcription results."""

    results: List[TranscriptionOutput]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    model: str
    adapter: str
    device: str


# ============================================================
# Model Loading
# ============================================================

class InferenceModel:
    """Wrapper for DisgraPhi model inference."""

    def __init__(self, adapter_path: str, model_name: str, device: str = "auto"):
        """
        Load model with LoRA adapter.

        Args:
            adapter_path: Path to LoRA adapter checkpoint
            model_name: Hugging Face model ID
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.adapter_path = adapter_path
        self.model_name = model_name

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing model on device: {self.device}")

        # Load model with adapter
        self.model = QwenVLHandwritingModel(
            model_name=model_name,
            load_in_4bit=True if self.device == "cuda" else False
        )

        # Load LoRA adapter
        logger.info(f"Loading adapter from: {adapter_path}")
        self.model.load_adapter(adapter_path)

        # Set to eval mode
        self.model.model.eval()

        logger.info("Model loaded successfully")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for inference with JPEG compression.

        Args:
            image: PIL Image

        Returns:
            Preprocessed PIL Image with compression applied
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply JPEG compression to reduce memory footprint
        # This helps with the GPU memory issue mentioned in line 217
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        compressed_image = Image.open(buffer)

        return compressed_image

    def transcribe(
        self,
        image: Image.Image,
        max_length: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        Transcribe handwritten text from image.

        Args:
            image: PIL Image containing handwriting
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Transcribed text
        """
        # Preprocess
        image = self._preprocess_image(image)

        # Create chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe this handwritten text."}
                ]
            }
        ]

        # Apply chat template with generation prompt
        text_prompt = self.model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.model.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        logger.info("About to transfer to device")

        # Move to device
        inputs = {k: v.to(self.model.model.device) for k, v in inputs.items()} # TODO: this is not great because if we have a big input it will not fit in the memory of the GPU - need to implement a more sequential way of loading the data because sometimes it would make us load 20GB
        logger.info("Transfered to device")

        # Generate
        with torch.no_grad():
            if temperature > 0:
                output_ids = self.model.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
            else:
                output_ids = self.model.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False
                )

        # Decode only generated tokens
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        transcription = self.model.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()


        # TODO: post process
        return transcription

    def transcribe_batch(
        self,
        images: List[Image.Image],
        max_length: int = 128,
        temperature: float = 0.0
    ) -> List[str]:
        """
        Transcribe multiple images in batch.

        Args:
            images: List of PIL Images
            max_length: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of transcribed texts
        """
        results = []

        # Process each image individually for now
        # (batching with dynamic image sizes is complex in Qwen2-VL)
        for image in images:
            try:
                text = self.transcribe(image, max_length, temperature)
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to transcribe image: {e}")
                results.append("")

        return results


# ============================================================
# Initialize Model
# ============================================================

logger.info("=" * 60)
logger.info("Initializing DisgraPhi Inference Server")
logger.info("=" * 60)

try:
    model = InferenceModel(
        adapter_path=ADAPTER_PATH,
        model_name=MODEL_NAME,
        device=DEVICE
    )
    logger.info("Model initialization complete")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="DisgraPhi Inference API",
    description="Handwriting recognition inference using Qwen2-VL with LoRA",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="disgraphi-inference",
        model=MODEL_NAME,
        adapter=ADAPTER_PATH,
        device=model.device
    )


@app.post("/transcribe", response_model=TranscriptionOutput)
def transcribe(data: ImageInput):
    """
    Transcribe handwritten text from a single image.

    Args:
        data: ImageInput with base64-encoded image

    Returns:
        TranscriptionOutput with transcribed text
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_bytes))

        # Run inference
        logger.info("Processing transcription request")
        text = model.transcribe(
            image=image,
            max_length=data.max_length,
            temperature=data.temperature
        )

        logger.info(f"Transcription complete: {len(text)} characters")

        return TranscriptionOutput(text=text, confidence=None)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@app.post("/transcribe/batch", response_model=BatchTranscriptionOutput)
def transcribe_batch(data: BatchImageInput):
    """
    Transcribe handwritten text from multiple images.

    Args:
        data: BatchImageInput with list of base64-encoded images

    Returns:
        BatchTranscriptionOutput with list of transcriptions
    """
    try:
        # Decode all images
        images = []
        for img_b64 in data.images:
            image_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

        # Run batch inference
        logger.info(f"Processing batch transcription request: {len(images)} images")
        texts = model.transcribe_batch(
            images=images,
            max_length=data.max_length,
            temperature=data.temperature
        )

        results = [TranscriptionOutput(text=text, confidence=None) for text in texts]

        logger.info(f"Batch transcription complete")

        return BatchTranscriptionOutput(results=results)

    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch transcription failed: {str(e)}"
        )


@app.post("/transcribe/file", response_model=TranscriptionOutput)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcribe handwritten text from an uploaded image file.

    Args:
        file: Uploaded image file

    Returns:
        TranscriptionOutput with transcribed text
    """
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        logger.info(f"Processing file upload: {file.filename}")
        text = model.transcribe(image=image)

        logger.info(f"File transcription complete: {len(text)} characters")

        return TranscriptionOutput(text=text, confidence=None)

    except Exception as e:
        logger.error(f"File transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File transcription failed: {str(e)}"
        )


# ============================================================
# Main (for local testing)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ML_INFERENCE_PORT", "8000"))

    logger.info(f"Starting server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
