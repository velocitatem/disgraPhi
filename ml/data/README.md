# DisgraPhi Data Pipeline

This directory contains the complete data processing pipeline for DisgraPhi handwriting recognition.

## Components

### 1. ETL Module (`etl.py`)

**IAMDownloader**: Downloads and processes IAM Handwriting Database
- Authenticated download with caching
- XML ground truth parsing
- Train/val/test splits by writer
- Line-level image organization

**PacketProcessor**: Processes user personalization packets
- QR corner detection for alignment
- Perspective transform (de-skewing)
- Line segmentation via horizontal projection
- Ground truth matching

### 2. Datasets Module (`datasets.py`)

**IAMDataset**: PyTorch dataset for bootstrap training
- 13k+ handwritten line images
- Pre-split train/val/test sets
- Grouped by writer for proper evaluation

**PersonalizationDataset**: PyTorch dataset for user-specific fine-tuning
- 40-120 user-written lines
- Auto-labeled from practice packets
- Few-shot learning format

### 3. CLI Interface (`data.py`)

Unified command-line interface for all data operations.

## Usage

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set IAM credentials (register at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database):
```bash
export IAM_USERNAME=your_username
export IAM_PASSWORD=your_password
```

### Download IAM Database

```bash
# Download and process IAM dataset
python ml/data/data.py download-iam

# Force re-download
python ml/data/data.py download-iam --force

# Custom location
python ml/data/data.py download-iam --data-dir /path/to/iam
```

Output structure:
```
ml/data/raw/iam/
├── processed/
│   ├── lines/          # Line images (a01-000u-00.png)
│   ├── ground_truth/   # Text files (a01-000u-00.txt)
│   └── splits/         # train.txt, val.txt, test.txt
└── raw/                # Original downloaded files
```

### Generate Personalization Packet

```bash
# Generate practice packet for a user
python ml/data/data.py generate-packet --user-id user001 --output packet.pdf

# Custom paths
python ml/data/data.py generate-packet \
    --user-id user001 \
    --output packets/user001.pdf \
    --ground-truth packets/user001_gt.json
```

The generated PDF contains:
- 8 pangram lines (alphabet coverage)
- Bigram sequences (common letter pairs)
- Numbers and dates
- Name/address placeholders
- QR codes at corners for alignment
- Line guides for writing

### Process Photographed Packet

After user fills out and photographs the packet:

```bash
# Process packet images
python ml/data/data.py process-packet \
    --user-id user001 \
    --images page1.jpg page2.jpg \
    --ground-truth packets/user001_gt.json

# Custom output location
python ml/data/data.py process-packet \
    --user-id user001 \
    --images page*.jpg \
    --ground-truth packets/user001_gt.json \
    --output-dir ml/data/users/user001
```

Output structure:
```
ml/data/users/user001/
├── lines/              # Cropped line images
│   ├── line_000.png
│   ├── line_001.png
│   └── ...
└── ground_truth.json   # Line ID to text mapping
```

### Validate Datasets

```bash
# Validate IAM dataset
python ml/data/data.py validate \
    --mode iam \
    --data-dir ml/data/raw/iam/processed \
    --show-sample

# Validate user dataset
python ml/data/data.py validate \
    --mode personalization \
    --user-dir ml/data/users/user001 \
    --show-sample
```

### List User Datasets

```bash
# List all processed users
python ml/data/data.py list-users

# Custom users directory
python ml/data/data.py list-users --users-dir /path/to/users
```

## Complete Workflow Example

```bash
# 1. Download IAM database (one-time setup)
python ml/data/data.py download-iam

# 2. Generate practice packet for user
python ml/data/data.py generate-packet --user-id alice --output alice_packet.pdf

# 3. User fills out packet and photographs it
# (Manual step: print, write, photograph)

# 4. Process photographed packet
python ml/data/data.py process-packet \
    --user-id alice \
    --images alice_page1.jpg alice_page2.jpg \
    --ground-truth alice_packet_ground_truth.json

# 5. Validate processed data
python ml/data/data.py validate --mode personalization --user-dir ml/data/users/alice

# 6. Ready for personalization training!
python ml/models/train.py --mode personalize --user-id alice
```

## Technical Details

### QR Code Alignment

The packet generator places QR codes at four corners:
- **TL**: Top-left
- **TR**: Top-right
- **BL**: Bottom-left
- **BR**: Bottom-right

The processor detects these codes and applies a perspective transform to correct for:
- Camera angle
- Page skew
- Lens distortion

### Line Segmentation

Line detection uses horizontal projection:
1. Convert to grayscale and binarize (Otsu's method)
2. Sum pixel intensities along horizontal axis
3. Find peaks (regions with significant ink)
4. Add padding and crop individual lines

### Ground Truth Matching

Current implementation: Sequential matching (line_000 → first expected text, etc.)

Future improvements:
- OCR-based confidence scoring
- Edit distance matching
- User verification interface

## File Formats

### Ground Truth JSON
```json
{
  "line_000": "The quick brown fox jumps over the lazy dog",
  "line_001": "Pack my box with five dozen liquor jugs",
  ...
}
```

### Split Files (train.txt, val.txt, test.txt)
```
a01-000u-00
a01-000u-01
a01-000u-02
...
```

## Caching Strategy

- IAM downloads are cached in `ml/data/raw/iam/raw/`
- Extraction markers prevent re-processing (`.{filename}.extracted`)
- Processed data is never regenerated unless forced
- Use `--force` flag to override caching

## Error Handling

Common issues:

1. **IAM authentication failure**: Check credentials
2. **QR codes not detected**: Ensure good lighting, no glare
3. **Line segmentation errors**: Verify line guides are visible
4. **Mismatched line counts**: Check for pages missing from photos

Enable debug mode for detailed errors:
```bash
DEBUG=1 python ml/data/data.py [command]
```

## Next Steps

After data pipeline setup:
1. Implement model architecture (`ml/models/arch.py`)
2. Create training scripts (`ml/models/train.py`)
3. Build inference API (`ml/inference.py`)
4. Develop Streamlit app (`apps/webapp-minimal/app.py`)
