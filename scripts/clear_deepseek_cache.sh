#!/bin/bash
# Clear DeepSeek-OCR cached model code that's incompatible with transformers 4.45+

echo "Clearing DeepSeek-OCR cached model code..."

CACHE_DIR="$HOME/.cache/huggingface/modules/transformers_modules"

# Remove all deepseek-related cached modules
if [ -d "$CACHE_DIR" ]; then
    echo "Found cache directory: $CACHE_DIR"

    # List what will be deleted
    find "$CACHE_DIR" -type d -iname "*deepseek*" -o -iname "*DeepSeek*"

    # Delete
    find "$CACHE_DIR" -type d \( -iname "*deepseek*" -o -iname "*DeepSeek*" \) -exec rm -rf {} + 2>/dev/null

    echo "✓ Cache cleared successfully"
    echo ""
    echo "Next steps:"
    echo "1. The model code will be re-downloaded on next training run"
    echo "2. If the issue persists, the upstream model may need updates"
else
    echo "Cache directory not found at: $CACHE_DIR"
fi
