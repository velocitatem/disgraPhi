"""
DisgraPhi Inference API

FastAPI server for handwriting recognition inference using trained LoRA adapters.

Environment Variables:
    ML_ADAPTER_PATH: Path to LoRA adapter checkpoint (required)
    ML_MODEL_NAME: Hugging Face model name (default: smolvlm-256m)
    ML_DEVICE: Device to run inference on (default: auto)
    ML_INFERENCE_PORT: Port to run server on (default: 8000)
    ML_INFERENCE_TIMEOUT: Request timeout in seconds (default: 300)

Timeout Configuration:
    The server is configured to handle long-running inference tasks without timing out.
    Default timeout is 300 seconds (5 minutes) which should be sufficient for most
    inference workloads including batch processing. Adjust ML_INFERENCE_TIMEOUT as needed.
"""

import os
import io
import base64
from typing import Optional, List
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from PIL import Image
from alveslib import get_logger, post_process_inference

from ml.models.providers import create_model


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
MODEL_NAME = os.getenv("ML_MODEL_NAME", "smolvlm-256m")
DEVICE = os.getenv("ML_DEVICE", "auto")
DEVICE = "cpu"

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
        self.model = create_model(model_name)
        # Load LoRA adapter
        logger.info(f"Loading adapter from: {adapter_path}")
        self.model.load_adapter(adapter_path)

        # Set to eval mode
        self.model.eval()

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

        logger.info("Transfered to device")

        # Generate
        with torch.no_grad():
            if temperature > 0:
                transcription = self.model.generate(
                    pixel_values=image,
                    prompt="Transcribe this handwritten text.",
                    max_new_tokens=max_length,
                    temperature=temperature,
                )
            else:
                transcription = self.model.generate(
                    pixel_values=image,
                    prompt="Transcribe this handwritten text.",
                    max_new_tokens=max_length,
                )

        print(f"BEFORE:\n{transcription}")
        # Post-process transcription for readability using the original image
        transcription = post_process_inference(transcription, image)
        print(f"AFTER:\n{transcription}")

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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Timeout middleware for long-running requests
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """
    Middleware to handle request timeouts gracefully.
    This ensures long-running inference tasks don't timeout prematurely.
    """
    try:
        # Get timeout from environment or use default
        timeout = int(os.getenv("ML_INFERENCE_TIMEOUT", "300"))

        # Process request with timeout
        response = await asyncio.wait_for(
            call_next(request),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {timeout}s: {request.url.path}")
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"Request timeout after {timeout} seconds. "
                          f"Consider increasing ML_INFERENCE_TIMEOUT or reducing batch size."
            }
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
        print(text)

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
    timeout = int(os.getenv("ML_INFERENCE_TIMEOUT", "300"))

    logger.info(f"Starting server on port {port}")
    logger.info(f"Request timeout set to {timeout}s")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=timeout,
        timeout_graceful_shutdown=30
    )
