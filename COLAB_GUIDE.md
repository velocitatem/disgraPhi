# DisgraPhi Google Colab Guide

## Quick Start with Colab

The easiest way to get started with DisgraPhi is through our Google Colab notebook. No installation required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/velocitatem/disgraPhi/blob/main/DisgraPhi_Personalization.ipynb)

Click the badge above to open the notebook directly in Google Colab.

## What You'll Do

1. **Setup** (5 minutes): Install DisgraPhi and dependencies
2. **Upload Samples** (5 minutes): Upload 10-30 images of your handwriting
3. **Label Data** (10 minutes): Type what each sample says
4. **Train Model** (15-30 minutes): Fine-tune a model on your handwriting
5. **Test & Download** (5 minutes): Try your model and download it

**Total time: ~45-60 minutes**

## Requirements

### Hardware
- **GPU Required**: Enable GPU in Colab (Runtime → Change runtime type → T4 GPU)
- Free tier is sufficient (no Colab Pro needed)

### Data You Need

#### Option 1: Quick Start (Minimal)
- 10-15 handwriting samples
- Each should be a clear photo of a single line of text
- Variety is key (different words/sentences)

#### Option 2: Best Results (Recommended)
- 30-50 handwriting samples
- Mix of:
  - Common words and phrases
  - Numbers and dates
  - Special characters if needed
  - Your typical writing scenarios

#### Option 3: Pro (Maximum Accuracy)
- 100+ samples
- Use the packet generator (see below)
- Professional-grade personalization

## Preparing Your Handwriting Samples

### Method 1: Use What You Have
1. Take photos of existing handwritten notes
2. Use your phone camera - no scanner needed
3. One line per image works best
4. Clear lighting, no shadows

### Method 2: Generate Practice Packet (Recommended)

For best results, use our packet generator to create structured practice sheets:

```bash
# Clone the repository
git clone https://github.com/velocitatem/disgraPhi.git
cd disgraPhi

# Install dependencies
pip install -r requirements.txt

# Generate practice packet
python ml/data/data.py generate-packet \
    --user-id your_name \
    --output practice_packet.pdf
```

This creates a PDF with:
- Pangrams (sentences using all letters)
- Common bigrams (letter pairs)
- Numbers and dates
- Practice lines with guides
- QR codes for automatic alignment

**Steps:**
1. Print the generated PDF
2. Fill it out with your normal handwriting
3. Take photos of each page
4. Use the packet processor (optional) or upload directly to Colab

### Method 3: Quick Handwriting Samples

Write these on paper and photograph:

```
The quick brown fox jumps over the lazy dog
Pack my box with five dozen liquor jugs
How vexingly quick daft zebras jump
Sphinx of black quartz, judge my vow
The five boxing wizards jump quickly
Today is [current date]
My name is [your name]
The time is [current time]
1234567890
!@#$%^&*()
```

## Tips for Best Results

### Photography
- ✅ Good lighting (natural light is best)
- ✅ Straight-on angle (not diagonal)
- ✅ High contrast (dark ink, white paper)
- ✅ Clear focus (no blur)
- ❌ Avoid shadows across text
- ❌ Avoid glare or reflections
- ❌ Don't use filters or edits

### Handwriting
- ✅ Write naturally (don't change your style)
- ✅ Use your normal pen/pencil
- ✅ Include variations (if your writing varies)
- ✅ Write at normal speed
- ❌ Don't write extra carefully
- ❌ Don't trace over letters
- ❌ Don't use all caps unless that's your normal style

### Labeling
- ✅ Type exactly what you wrote (including mistakes!)
- ✅ Preserve punctuation and spacing
- ✅ Use the same capitalization
- ❌ Don't "fix" what you wrote
- ❌ Don't guess - verify each label

## Training Configuration

### Small Dataset (10-20 samples)
The notebook automatically adjusts:
- 10 epochs
- Strong augmentation
- Lower learning rate

### Medium Dataset (20-40 samples)  
- 5 epochs (default)
- Moderate augmentation
- Standard learning rate

### Large Dataset (40+ samples)
- 3 epochs
- Light augmentation
- Higher learning rate

You can override these in the notebook if needed.

## Expected Results

### Accuracy Metrics

With proper samples, expect:

| Samples | Character Error Rate (CER) | Word Error Rate (WER) |
|---------|---------------------------|----------------------|
| 10-15   | 15-25%                    | 25-40%              |
| 20-30   | 8-15%                     | 15-25%              |
| 40-60   | 5-10%                     | 10-18%              |
| 100+    | 2-5%                      | 5-12%               |

*Lower is better*

### Real-World Performance

- **Legible handwriting**: Excellent results even with 15-20 samples
- **Challenging handwriting**: 40+ samples recommended
- **Dysgraphia/motor difficulties**: 60+ samples for best personalization

## Using Your Trained Model

### In Colab
The notebook includes testing cells - just upload new handwriting!

### Download for Local Use

After training, you'll download a `.zip` file containing:
- `adapter/` - Your personalized LoRA adapter
- `model_info.json` - Training details
- `README.md` - Usage instructions

### Load Locally

```python
from ml.models.providers import create_model

# Load base model
model = create_model("smolvlm-256m")

# Load your personalized adapter
model.load_adapter("path/to/adapter")

# Transcribe image
from PIL import Image
img = Image.open("my_handwriting.jpg")

text = model.generate(
    pixel_values=img,
    prompt="Transcribe this handwritten text.",
    max_new_tokens=128
)

print(text)
```

### Deploy as API

Use the included inference server:

```bash
# Set environment variables
export ML_ADAPTER_PATH="path/to/adapter"
export ML_MODEL_NAME="smolvlm-256m"

# Start server
python ml/inference.py
```

API will be available at `http://localhost:8000`

## Troubleshooting

### "No GPU detected"
- Go to Runtime → Change runtime type
- Set Hardware accelerator to "GPU"
- Choose T4 GPU (free tier)

### "Out of memory" during training
- Reduce batch size in the notebook
- Reduce number of epochs
- Use fewer/smaller images
- Restart runtime and try again

### Poor accuracy
- Add more samples (aim for 30+)
- Check labeling accuracy
- Include more variety in samples
- Try training for more epochs
- Check image quality

### Training is very slow
- Confirm GPU is enabled
- Check runtime type (should show GPU RAM)
- T4 GPU should complete in 15-30 minutes

### Model doesn't recognize handwriting
- Ensure samples match your target handwriting style
- Check that labels are accurate
- Add more samples
- Consider using the packet generator for structured data

## Advanced Usage

### Custom Training Parameters

Edit the training cell to adjust:

```python
# More aggressive training
num_epochs = 10
learning_rate = 5e-5
batch_size = 1

# Conservative training (less overfitting)
num_epochs = 3
learning_rate = 1e-5
batch_size = 4
```

### Using Pre-trained Bootstrap Model

If available, use a bootstrap model trained on IAM dataset:

```python
!python ml/models/train.py \
    --bootstrap_adapter_path "path/to/bootstrap/adapter" \
    ... # other parameters
```

This improves results with fewer samples.

### Batch Processing

Process multiple users' models:

1. Prepare separate folders for each user
2. Run notebook for each user
3. Store adapters with user IDs
4. Deploy multi-adapter inference server

## Getting Help

### Common Questions
- **Q: How long does training take?**  
  A: 15-30 minutes on Colab's free T4 GPU

- **Q: Can I use my own dataset format?**  
  A: Yes! See `ml/data/datasets.py` for ManifestDataset format

- **Q: Do I need coding experience?**  
  A: No! Just click cells in order and follow prompts

- **Q: Can I improve an existing model?**  
  A: Yes! Add more samples and retrain

- **Q: What if my handwriting is messy?**  
  A: That's exactly what DisgraPhi is for! More samples help

### Support Channels
- 📖 [Full Documentation](https://github.com/velocitatem/disgraPhi)
- 🐛 [Report Issues](https://github.com/velocitatem/disgraPhi/issues)
- 💬 [Discussions](https://github.com/velocitatem/disgraPhi/discussions)

## Contributing

Improve the Colab experience:
- Test with different handwriting styles
- Report bugs or unclear instructions
- Suggest improvements
- Share your results (anonymously)

## License

DisgraPhi is open source. Your trained models are yours to keep and use however you like.

---

**Ready to get started?** Click the badge at the top of this file to open in Colab!
