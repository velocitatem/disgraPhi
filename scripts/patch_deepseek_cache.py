#!/usr/bin/env python3
"""
Patch cached DeepSeek-OCR model code for transformers 4.45+ compatibility.

The upstream DeepSeek-OCR model code imports LlamaFlashAttention2 which was
removed in transformers 4.45+. This script patches the cached model code to
work with newer transformers versions.
"""

import os
import re
from pathlib import Path

def find_deepseek_cache():
    """Find all cached DeepSeek-OCR model directories."""
    cache_base = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"

    if not cache_base.exists():
        print(f"Cache directory not found: {cache_base}")
        return []

    # Find all DeepSeek directories
    deepseek_dirs = []
    for item in cache_base.rglob("*"):
        if item.is_dir() and "deepseek" in item.name.lower():
            deepseek_dirs.append(item)

    return deepseek_dirs

def patch_modeling_file(file_path):
    """Patch modeling_deepseekv2.py to remove LlamaFlashAttention2 import."""
    if not file_path.exists():
        return False

    print(f"Patching: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED FOR TRANSFORMERS 4.45+" in content:
        print("  ✓ Already patched")
        return True

    original_content = content

    # Pattern 1: Remove LlamaFlashAttention2 from imports
    content = re.sub(
        r'LlamaFlashAttention2,?\s*',
        '',
        content
    )

    # Pattern 2: Remove entire flash attention import line if it's standalone
    content = re.sub(
        r'from transformers\.models\.llama\.modeling_llama import LlamaFlashAttention2\n',
        '# PATCHED FOR TRANSFORMERS 4.45+: Removed LlamaFlashAttention2 import\n',
        content
    )

    # Pattern 3: Clean up any remaining references in LLAMA_ATTENTION_CLASSES dict
    content = re.sub(
        r'"flash_attention_2":\s*LlamaFlashAttention2,?\s*',
        '',
        content
    )

    if content != original_content:
        # Backup original
        backup_path = file_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"  ✓ Backup created: {backup_path}")

        # Write patched version
        with open(file_path, 'w') as f:
            f.write(content)
        print("  ✓ Patched successfully")
        return True
    else:
        print("  - No changes needed")
        return False

def main():
    print("=" * 80)
    print("DeepSeek-OCR Cache Patcher")
    print("=" * 80)
    print()

    # Find cached directories
    deepseek_dirs = find_deepseek_cache()

    if not deepseek_dirs:
        print("No DeepSeek cached directories found.")
        print()
        print("If you haven't downloaded the model yet, the patching will happen")
        print("automatically after the first download attempt.")
        return

    print(f"Found {len(deepseek_dirs)} DeepSeek cache director{'y' if len(deepseek_dirs) == 1 else 'ies'}:")
    for d in deepseek_dirs:
        print(f"  - {d}")
    print()

    # Patch each directory
    patched_count = 0
    for cache_dir in deepseek_dirs:
        modeling_file = cache_dir / "modeling_deepseekv2.py"
        if modeling_file.exists():
            if patch_modeling_file(modeling_file):
                patched_count += 1

    print()
    print("=" * 80)
    print(f"Patching complete: {patched_count} file(s) patched")
    print("=" * 80)
    print()
    print("You can now run training with:")
    print("  make train-quick-test")

if __name__ == "__main__":
    main()
