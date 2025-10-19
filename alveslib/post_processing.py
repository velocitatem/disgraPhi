from openai import OpenAI
import base64
import io
from PIL import Image

def post_process_inference(transcription: str, image: Image.Image) -> str:
    """
    Post-process the transcription using OpenAI's language model to enhance readability.

    Args:
        transcription (str): The raw transcription text.
        image (Image.Image): PIL Image object to provide visual context.

    Returns:
        str: The post-processed transcription text.
    """
    client = OpenAI()

    prompt = f"""Attempted transcription: <TRANSCRIPTION>{transcription}</TRANSCRIPTION> Of a person with dysgraphia. Try to return a proper full transcription accounting for their disability by trying to use the context relevant. Return transcription wrapped properly in XML tags like above."""

    def encode_image_from_pil(pil_image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Save as JPEG to buffer
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    # Encode the PIL image
    base64_image = encode_image_from_pil(image)

    # Compose prompt and image input
    response = client.responses.create(
        model="gpt-5",  # or the model you want to use; adjust per API docs
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]
    )

    text = response.output_text.strip()
    start_position = text.find("<TRANSCRIPTION>")
    end_position = text.find("</TRANSCRIPTION>") + len("</TRANSCRIPTION>")

    if start_position != -1 and end_position != -1:
        # Extract content between tags
        extracted = text[start_position + len("<TRANSCRIPTION>"):end_position - len("</TRANSCRIPTION>")]
        return extracted.strip()

    return response.output_text

if __name__ == "__main__":
    sample_transcription = "• FRANCIS: HOW ARE THE CHOICES PRESENTED?\n• ARCHON: WHAT INFO IS PRESENTED? WHO IS THE DOCUMENT? IS THERE A FACTOR OF CANCELLATION? POSITION IN CANCELLATION? THEN WHAT? VACUE"
    image_path = "/home/velocitatem/Downloads/signal-2025-10-18-172601.jpeg"  # Replace with your actual image path

    # Load image as PIL Image
    test_image = Image.open(image_path)
    processed_text = post_process_inference(sample_transcription, test_image)
    print("Post-processed Transcription:\n", processed_text)
