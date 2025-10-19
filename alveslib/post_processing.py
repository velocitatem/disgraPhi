from openai import OpenAI
import base64

def post_process_inference(transcription : str, image_path : str) -> str:
    """
    Post-process the transcription using OpenAI's language model to enhance readability.
    Args:
        transcription (str): The raw transcription text.
    Returns:
        str: The post-processed transcription text.
    """
    client = OpenAI()

    prompt = f"""Attempted transcription: <TRANSCRIPTION>{transcription}</TRANSCRIPTION> Of a person with dysgraphie. Try to return a proper full transcription accounting for their disability by trying to use the context relevant. Return transcription wrapped properly in XML tags like above."""
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image file
    base64_image = encode_image(image_path)

    # Compose prompt and image input
    response = client.responses.create(
        model="gpt-5", # or the model you want to use; adjust per API docs
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
        return text[start_position:end_position]

    return response.output_text

if __name__ == "__main__":
    sample_transcription = "• FRANCIS: HOW ARE THE CHOICES PRESENTED?\n• ARCHON: WHAT INFO IS PRESENTED? WHO IS THE DOCUMENT? IS THERE A FACTOR OF CANCELLATION? POSITION IN CANCELLATION? THEN WHAT? VACUE"
    image_path = "/home/velocitatem/Downloads/signal-2025-10-18-172601.jpeg"  # Replace with your actual image path
    processed_text = post_process_inference(sample_transcription, image_path)
    print("Post-processed Transcription:\n", processed_text)
