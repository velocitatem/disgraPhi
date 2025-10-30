# Handwriting Data Collector (Gradio)

Interactive webapp for collecting high-quality handwriting samples for DisgraPhi personalization.

## Quick Start

```bash
# Install dependencies
pip install gradio>=4.0 gradio_image_annotation>=0.4 pillow

# Run the collector
python app.py
```

Opens in your browser at `http://localhost:7860`

## Features

- **Guided Data Collection**: Prompts you with 10 practice sentences
  - Pangrams (all letters): "The quick brown fox jumps over the lazy dog"
  - Numbers: "0 1 2 3 4 5 6 7 8 9"
  - Symbols: "@ # $ % & * ( )"
  - Common phrases and email/address formats

- **Flexible Input**: 
  - Upload photos from your phone
  - Use webcam to capture in real-time
  - Paste images from clipboard

- **Interactive Annotation**: Draw bounding boxes around your handwriting with visual guidance

- **Automatic Processing**: 
  - Crops each sentence to just the handwriting region
  - Packages everything into a ready-to-use ZIP file
  - Generates manifest.json with all metadata

- **Quality Validation**: See thumbnails of all captured samples before exporting

## How It Works

1. **Write**: Write each prompted sentence on paper in your normal handwriting
2. **Photograph**: Take a photo using the webapp (upload/webcam/paste)
3. **Annotate**: Draw a tight box around the sentence
4. **Next**: Click "Save and next" to move to the next sentence
5. **Export**: After all 10 sentences, download the ZIP file

## Output Format

The exported ZIP contains:

```
disgraf_XXXXXXXX/
├── images/           # Original photos
│   ├── 00.png
│   ├── 01.png
│   └── ...
├── crops/            # Cropped handwriting regions
│   ├── 00.png
│   ├── 01.png
│   └── ...
└── annotations.jsonl # Metadata (bbox, text, timestamp)
```

## Using with DisgraPhi

### Option 1: Colab Notebook

1. Run this webapp and download the ZIP
2. Extract the ZIP file
3. Open [DisgraPhi Colab Notebook](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)
4. Upload the `crops/*.png` files
5. In the labeling step, use the sentences from `annotations.jsonl`

### Option 2: Local Training

```bash
# Extract the downloaded ZIP
unzip disgraf_XXXXXXXX.zip -d my_handwriting

# Convert to manifest format (if needed)
# The annotations.jsonl needs to be converted to manifest.json format

# Train
disgraphi-train \
    --dataset_type manifest \
    --manifest_data_dir my_handwriting/crops \
    --num_train_epochs 5
```

## Tips for Best Results

### Photography
- ✅ Good, even lighting (natural light is best)
- ✅ Camera parallel to the paper
- ✅ Paper flat, not curved
- ✅ No shadows across the text
- ❌ Avoid glare or reflections
- ❌ Don't use filters or edits

### Annotation
- ✅ Draw box tight around the text (include all ink)
- ✅ Include ascenders (tall letters like 'h', 'l') and descenders (letters like 'g', 'y')
- ❌ Don't leave too much white space
- ❌ Don't cut off parts of letters

### Handwriting
- ✅ Write naturally (don't try to be extra neat)
- ✅ Use your normal pen/pencil
- ✅ Write at your normal speed
- ❌ Don't trace or write extra slowly
- ❌ Don't change your style mid-collection

## Customization

Edit the `SENTENCES` list in `app.py` to use different practice text:

```python
SENTENCES = [
    "Your custom sentence 1",
    "Your custom sentence 2",
    # ... add more
]
```

For different languages, update the sentences to include characters specific to that language.

## Requirements

- Python 3.10+
- gradio >= 4.0
- gradio_image_annotation >= 0.4
- Pillow

## Troubleshooting

**Webapp won't start:**
```bash
pip install --upgrade gradio gradio_image_annotation
```

**Can't draw boxes:**
- Make sure you uploaded an image first
- Try refreshing the page
- Check browser console for errors

**Webcam not working:**
- Grant camera permissions to your browser
- Try using upload instead
- Check if another app is using the webcam

## Why This Instead of Packet Generation?

The previous approach used PDF generation with QR codes for alignment, but had issues:
- ❌ Requires printing
- ❌ QR code detection can fail
- ❌ Multiple manual steps
- ❌ No immediate feedback

This Gradio approach is better:
- ✅ No printing needed
- ✅ Interactive with immediate visual feedback
- ✅ Works on any device with a camera
- ✅ Automatic quality validation
- ✅ Simpler workflow

## License

Part of DisgraPhi - MIT License
