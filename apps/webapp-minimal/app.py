# requirements:
#   gradio>=4.0
#   pillow
#   numpy

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import uuid
import tempfile
import os
import json
import shutil
import datetime as dt
from pathlib import Path

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

# Default practice sentences (user can add more)
DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please write this sentence in your normal handwriting.",
    "Numbers: 0 1 2 3 4 5 6 7 8 9",
    "Uppercase: ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "Lowercase: abcdefghijklmnopqrstuvwxyz",
    "Email example: name@example.com",
    "Address: 221B Baker Street, London",
    "Punctuation: ! @ # $ % ^ & * ( ) - _ = +",
    "Special chars: [ ] { } | \\ / < > ? .",
    "Mixed case: The Rain in Spain Falls Mainly on the Plain",
]

def _new_session():
    """Initialize a new data collection session."""
    sid = str(uuid.uuid4())[:8]
    root = os.path.join(tempfile.gettempdir(), f"disgraphi_data_{sid}")
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    return {
        "sid": sid,
        "root": root,
        "items": [],
        "current_index": 0
    }

def _create_manifest(state, split_ratios=(0.8, 0.1, 0.1)):
    """
    Create manifest.json in the format expected by DisgraPhi training.

    Format:
    {
        "train": [{"image": "path/to/image.png", "text": "ground truth"}],
        "val": [...],
        "test": [...]
    }
    """
    items = state["items"]
    n = len(items)

    # Calculate split indices
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    manifest = {
        "train": [],
        "val": [],
        "test": []
    }

    for idx, item in enumerate(items):
        entry = {
            "image": item["image_path"],
            "text": item["text"]
        }

        if idx < train_end:
            manifest["train"].append(entry)
        elif idx < val_end:
            manifest["val"].append(entry)
        else:
            manifest["test"].append(entry)

    return manifest

def _save_drawing(drawing, state, text):
    """Save a drawing from the canvas."""
    if drawing is None:
        return None, "Please draw something first"

    # Convert to PIL Image
    if isinstance(drawing, dict) and 'composite' in drawing:
        # Gradio Sketchpad format
        img = Image.fromarray(drawing['composite'])
    elif isinstance(drawing, np.ndarray):
        img = Image.fromarray(drawing)
    else:
        img = drawing

    # Convert to grayscale and invert (white background, black text)
    img = img.convert('L')
    img = Image.eval(img, lambda x: 255 - x)

    # Crop to content (remove excess white space)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        # Add small padding
        padding = 10
        padded = Image.new('L', (img.width + padding*2, img.height + padding*2), 255)
        padded.paste(img, (padding, padding))
        img = padded

    return img, None

def _save_photo(photo, state, text):
    """Save an uploaded photo."""
    if photo is None:
        return None, "Please upload a photo first"

    # Convert to PIL Image
    if isinstance(photo, np.ndarray):
        img = Image.fromarray(photo)
    else:
        img = photo

    # Convert to RGB
    img = img.convert('RGB')

    return img, None

def save_sample(input_image, input_text, input_mode, state):
    """Save a single handwriting sample (drawing or photo)."""
    if not input_text or not input_text.strip():
        return (
            state,
            gr.update(),
            gr.update(),
            "⚠️ Please enter the text label before saving",
            _get_gallery(state),
            _get_stats(state)
        )

    text = input_text.strip()

    # Save based on input mode
    if input_mode == "draw":
        img, error = _save_drawing(input_image, state, text)
    else:  # photo
        img, error = _save_photo(input_image, state, text)

    if error:
        return (
            state,
            gr.update(),
            gr.update(),
            f"⚠️ {error}",
            _get_gallery(state),
            _get_stats(state)
        )

    # Save image
    idx = len(state["items"])
    filename = f"sample_{idx:04d}.png"
    filepath = os.path.join(state["root"], "images", filename)
    img.save(filepath)

    # Add to state
    state["items"].append({
        "index": idx,
        "text": text,
        "image_path": f"images/{filename}",
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "mode": input_mode
    })

    # Clear inputs for next sample
    status = f"✅ Sample {idx + 1} saved: '{text[:50]}{'...' if len(text) > 50 else ''}'"

    return (
        state,
        None,  # Clear canvas/photo
        "",    # Clear text input
        status,
        _get_gallery(state),
        _get_stats(state)
    )

def use_suggested_text(suggested_text, state):
    """Use a suggested sentence."""
    return suggested_text

def add_custom_sentence(custom_sentence, suggested_sentences):
    """Add a custom sentence to the suggestions."""
    if not custom_sentence or not custom_sentence.strip():
        return gr.update(), "⚠️ Please enter a sentence first"

    updated = suggested_sentences + [custom_sentence.strip()]
    return (
        gr.update(choices=updated, value=custom_sentence.strip()),
        f"✅ Added: '{custom_sentence.strip()}'"
    )

def _get_gallery(state):
    """Get gallery items for display."""
    items = []
    for item in state["items"]:
        img_path = os.path.join(state["root"], item["image_path"])
        caption = f"#{item['index'] + 1}: {item['text'][:50]}"
        items.append((img_path, caption))
    return items

def _get_stats(state):
    """Get collection statistics."""
    n = len(state["items"])
    if n == 0:
        return "No samples collected yet"

    draw_count = sum(1 for item in state["items"] if item["mode"] == "draw")
    photo_count = n - draw_count

    return f"""
### Collection Statistics
- **Total Samples**: {n}
- **Drawn**: {draw_count}
- **Photographed**: {photo_count}
- **Train/Val/Test Split**: {int(n*0.8)}/{int(n*0.1)}/{n - int(n*0.8) - int(n*0.1)}

**Recommendation**: Collect 50-200 samples for good personalization results.
"""

def delete_last(state):
    """Delete the last saved sample."""
    if not state["items"]:
        return (
            state,
            "⚠️ No samples to delete",
            _get_gallery(state),
            _get_stats(state)
        )

    # Remove last item
    item = state["items"].pop()

    # Delete file
    filepath = os.path.join(state["root"], item["image_path"])
    if os.path.exists(filepath):
        os.remove(filepath)

    return (
        state,
        f"✅ Deleted sample #{item['index'] + 1}",
        _get_gallery(state),
        _get_stats(state)
    )

def finalize_dataset(state, dataset_name, split_train, split_val):
    """Create final dataset with manifest.json."""
    if not state["items"]:
        return (
            state,
            gr.update(visible=False),
            "⚠️ No samples to package. Collect some samples first!",
        )

    # Validate split ratios
    split_test = 100 - split_train - split_val
    if split_test < 0:
        return (
            state,
            gr.update(visible=False),
            "⚠️ Split ratios must sum to ≤100%",
        )

    split_ratios = (split_train/100, split_val/100, split_test/100)

    # Create manifest
    manifest = _create_manifest(state, split_ratios)
    manifest_path = os.path.join(state["root"], "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save metadata
    metadata = {
        "dataset_name": dataset_name or f"handwriting_{state['sid']}",
        "created": dt.datetime.utcnow().isoformat() + "Z",
        "total_samples": len(state["items"]),
        "train_samples": len(manifest["train"]),
        "val_samples": len(manifest["val"]),
        "test_samples": len(manifest["test"]),
        "split_ratios": split_ratios
    }
    with open(os.path.join(state["root"], "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme = f"""# {metadata['dataset_name']}

Handwriting dataset for DisgraPhi personalization training.

## Statistics
- Total samples: {metadata['total_samples']}
- Train: {metadata['train_samples']}
- Val: {metadata['val_samples']}
- Test: {metadata['test_samples']}

## Usage

```bash
python ml/models/train_trl.py \\
  --model_provider deepseek-ocr \\
  --dataset_type manifest \\
  --manifest_data_dir /path/to/this/dataset \\
  --bootstrap_adapter_path /path/to/bootstrap/adapter \\
  --num_train_epochs 2 \\
  --augment \\
  --augment_strength 0.7
```

Or with Makefile:

```bash
make train-personalize \\
  MANIFEST_DIR=/path/to/this/dataset \\
  BOOTSTRAP_ADAPTER=/path/to/bootstrap/adapter
```

## Format

The `manifest.json` contains:
```json
{{
  "train": [{{"image": "images/sample_0000.png", "text": "ground truth"}}],
  "val": [...],
  "test": [...]
}}
```

Created: {metadata['created']}
"""
    with open(os.path.join(state["root"], "README.md"), 'w') as f:
        f.write(readme)

    # Create ZIP
    dataset_name_clean = (dataset_name or f"handwriting_{state['sid']}").replace(" ", "_")
    zip_path = shutil.make_archive(
        os.path.join(tempfile.gettempdir(), dataset_name_clean),
        "zip",
        state["root"]
    )

    success_msg = f"""
### ✅ Dataset Ready!

**{metadata['dataset_name']}**
- {metadata['total_samples']} total samples
- Split: {metadata['train_samples']} train / {metadata['val_samples']} val / {metadata['test_samples']} test

Download the ZIP file below. It contains:
- `images/` - All handwriting samples
- `manifest.json` - Training manifest
- `metadata.json` - Dataset info
- `README.md` - Usage instructions

Extract the ZIP and use the directory path for training:
```bash
python ml/models/train_trl.py \\
  --dataset_type manifest \\
  --manifest_data_dir /path/to/extracted/dataset
```
"""

    return (
        state,
        gr.update(visible=True, value=zip_path),
        success_msg,
    )

def restart_session():
    """Start a new session."""
    state = _new_session()
    return (
        state,
        None,  # Clear canvas
        "",    # Clear text
        "🔄 New session started. Ready to collect samples!",
        _get_gallery(state),
        _get_stats(state),
        gr.update(visible=False)
    )


# Build the Gradio interface
with gr.Blocks(title="DisgraPhi Handwriting Collector", theme=THEME) as demo:
    gr.Markdown("""
    # 📝 DisgraPhi Handwriting Collector

    Collect handwriting samples for personalized OCR training.

    **Two modes:**
    1. **Draw**: Write directly with mouse/stylus
    2. **Photo**: Upload pictures of handwritten text

    **Steps:**
    1. Choose input mode (Draw or Photo)
    2. Create/upload your handwriting sample
    3. Enter the exact text you wrote
    4. Click "Save Sample"
    5. Repeat 50-200 times for best results
    6. Click "Finalize Dataset" when done
    """)

    state = gr.State(_new_session())

    with gr.Row():
        with gr.Column(scale=2):
            # Input mode selector
            input_mode = gr.Radio(
                choices=["draw", "photo"],
                value="draw",
                label="Input Mode",
                info="Choose how you want to input handwriting"
            )

            # Dynamic input (changes based on mode)
            with gr.Group():
                # Drawing canvas
                canvas = gr.Sketchpad(
                    label="Draw your handwriting here",
                    type="numpy",
                    brush=gr.Brush(colors=["#000000"], default_size=3),
                    height=300,
                    visible=True
                )

                # Photo upload
                photo = gr.Image(
                    label="Upload a photo of handwriting",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    height=300,
                    visible=False
                )

            # Text input
            text_input = gr.Textbox(
                label="Text Label",
                placeholder="Enter the exact text you wrote...",
                lines=2,
                info="Type exactly what you wrote in the image"
            )

            # Action buttons
            with gr.Row():
                save_btn = gr.Button("💾 Save Sample", variant="primary", scale=2)
                delete_btn = gr.Button("🗑️ Delete Last", scale=1)

        with gr.Column(scale=1):
            # Suggested sentences
            gr.Markdown("### 📋 Suggested Sentences")
            suggested_dropdown = gr.Dropdown(
                choices=DEFAULT_SENTENCES,
                label="Quick Select",
                info="Click to use a suggested sentence"
            )
            use_suggested_btn = gr.Button("Use Selected Text", size="sm")

            with gr.Accordion("Add Custom Sentence", open=False):
                custom_sentence = gr.Textbox(
                    label="Custom Sentence",
                    placeholder="Type a custom sentence..."
                )
                add_custom_btn = gr.Button("Add to Suggestions", size="sm")
                custom_status = gr.Markdown("")

            # Statistics
            stats = gr.Markdown(_get_stats(state.value))

            # Status
            status = gr.Markdown("🔄 Ready to collect samples")

    # Gallery of collected samples
    with gr.Accordion("📸 Collected Samples", open=True):
        gallery = gr.Gallery(
            label="Your Handwriting Samples",
            columns=5,
            rows=2,
            height="auto",
            object_fit="contain"
        )

    # Finalization section
    with gr.Accordion("📦 Finalize Dataset", open=False):
        gr.Markdown("""
        ### Create Training Dataset

        When you've collected enough samples (50-200 recommended), create your training dataset.
        This will generate a `manifest.json` file compatible with DisgraPhi training.
        """)

        dataset_name = gr.Textbox(
            label="Dataset Name",
            placeholder="my_handwriting",
            value=f"handwriting_{dt.datetime.now().strftime('%Y%m%d')}"
        )

        with gr.Row():
            split_train = gr.Slider(
                minimum=50,
                maximum=90,
                value=80,
                step=5,
                label="Train Split (%)",
                info="Percentage for training"
            )
            split_val = gr.Slider(
                minimum=5,
                maximum=25,
                value=10,
                step=5,
                label="Val Split (%)",
                info="Percentage for validation"
            )

        finalize_btn = gr.Button("📦 Create Dataset Package", variant="primary")
        finalize_status = gr.Markdown("")
        download_file = gr.File(label="Download Dataset (.zip)", visible=False)

    # Restart button
    with gr.Row():
        restart_btn = gr.Button("🔄 Start New Collection", variant="secondary")

    # Event handlers
    def update_input_visibility(mode):
        """Toggle between canvas and photo upload."""
        return (
            gr.update(visible=(mode == "draw")),
            gr.update(visible=(mode == "photo"))
        )

    input_mode.change(
        update_input_visibility,
        inputs=[input_mode],
        outputs=[canvas, photo]
    )

    def save_with_mode(canvas_img, photo_img, text, mode, state):
        """Save sample based on selected mode."""
        img = canvas_img if mode == "draw" else photo_img
        return save_sample(img, text, mode, state)

    save_btn.click(
        save_with_mode,
        inputs=[canvas, photo, text_input, input_mode, state],
        outputs=[state, canvas, text_input, status, gallery, stats]
    )

    delete_btn.click(
        delete_last,
        inputs=[state],
        outputs=[state, status, gallery, stats]
    )

    use_suggested_btn.click(
        use_suggested_text,
        inputs=[suggested_dropdown, state],
        outputs=[text_input]
    )

    add_custom_btn.click(
        add_custom_sentence,
        inputs=[custom_sentence, suggested_dropdown],
        outputs=[suggested_dropdown, custom_status]
    )

    finalize_btn.click(
        finalize_dataset,
        inputs=[state, dataset_name, split_train, split_val],
        outputs=[state, download_file, finalize_status]
    )

    restart_btn.click(
        restart_session,
        outputs=[state, canvas, text_input, status, gallery, stats, download_file]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
