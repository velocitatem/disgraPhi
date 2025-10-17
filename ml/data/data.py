"""
DisgraPhi Data Pipeline CLI

Unified interface for data operations:
- Download and process IAM database
- Generate personalization packets
- Process photographed packets
- Validate datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.etl import IAMDownloader, PacketProcessor
from ml.data.datasets import IAMDataset, PersonalizationDataset
from src.packet_generator import PacketGenerator


def download_iam(args):
    """Download and process IAM Handwriting Database."""
    print("=" * 60)
    print("IAM Database Download & Processing")
    print("=" * 60)

    downloader = IAMDownloader(
        data_dir=args.data_dir,
        hf_token=args.hf_token,
        repo_id=getattr(args, 'repo_id', None)
    )

    # Download
    print("\n[1/2] Downloading IAM database...")
    downloader.download(force=args.force)

    # Process
    print("\n[2/2] Processing IAM data...")
    downloader.process()

    print("\n✓ IAM database ready for training")
    print(f"  Location: {downloader.processed_dir}")


def generate_packet(args):
    """Generate personalization practice packet."""
    print("=" * 60)
    print("Personalization Packet Generation")
    print("=" * 60)

    generator = PacketGenerator(user_id=args.user_id)

    print(f"\nGenerating packet for user: {args.user_id}")
    ground_truth = generator.generate(args.output)

    # Save ground truth
    gt_path = args.ground_truth or args.output.replace('.pdf', '_ground_truth.json')
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Packet generated successfully")
    print(f"  PDF: {args.output}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Total lines: {len(ground_truth)}")
    print(f"\nNext steps:")
    print(f"  1. Print {args.output}")
    print(f"  2. Fill out all lines in your handwriting")
    print(f"  3. Photograph each page")
    print(f"  4. Run: python ml/data/data.py process-packet --user-id {args.user_id} --images <paths>")


def process_packet(args):
    """Process photographed personalization packet."""
    print("=" * 60)
    print("Packet Processing")
    print("=" * 60)

    # Validate image paths
    image_paths = []
    for path in args.images:
        if not Path(path).exists():
            print(f"Warning: Image not found: {path}")
        else:
            image_paths.append(path)

    if not image_paths:
        print("Error: No valid image paths provided")
        return

    # Load ground truth
    gt_path = args.ground_truth
    if not gt_path:
        print("Error: Ground truth file required (use --ground-truth)")
        return

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    # Setup output directory
    user_dir = args.output_dir or f"ml/data/users/{args.user_id}"

    print(f"\nProcessing {len(image_paths)} images for user: {args.user_id}")
    print(f"Output directory: {user_dir}")

    # Process
    processor = PacketProcessor(user_dir=user_dir)
    results = processor.process_packet(
        image_paths=image_paths,
        ground_truth=ground_truth
    )

    print(f"\n✓ Packet processing complete")
    print(f"  Lines detected: {results['lines_detected']}")
    print(f"  Lines matched: {results['lines_matched']}")
    print(f"  Output: {results['output_dir']}")

    if results['lines_matched'] < len(ground_truth):
        print(f"\n⚠ Warning: Only {results['lines_matched']}/{len(ground_truth)} lines matched")
        print("  Check image quality and QR code visibility")
    else:
        print(f"\n✓ All lines matched successfully")
        print(f"\nNext steps:")
        print(f"  Run personalization training:")
        print(f"  python ml/models/train.py --mode personalize --user-id {args.user_id}")


def validate_dataset(args):
    """Validate dataset and show statistics."""
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    if args.mode == 'iam':
        print(f"\nValidating IAM dataset: {args.data_dir}")

        splits = ['train', 'val', 'test']
        for split in splits:
            try:
                dataset = IAMDataset(
                    data_dir=args.data_dir,
                    split=split
                )
                print(f"  {split:5s}: {len(dataset):6d} samples")

                # Show sample
                if len(dataset) > 0 and args.show_sample:
                    sample = dataset[0]
                    print(f"         Sample text: {sample['text'][:50]}...")

            except Exception as e:
                print(f"  {split:5s}: Error - {e}")

    elif args.mode == 'personalization':
        print(f"\nValidating personalization dataset: {args.user_dir}")

        try:
            dataset = PersonalizationDataset(user_dir=args.user_dir)
            print(f"  Total samples: {len(dataset)}")

            # Show samples
            if len(dataset) > 0 and args.show_sample:
                print(f"\n  Sample lines:")
                for i in range(min(5, len(dataset))):
                    sample = dataset[i]
                    print(f"    {sample['line_id']}: {sample['text'][:40]}...")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n✓ Validation complete")


def list_users(args):
    """List available user datasets."""
    print("=" * 60)
    print("Available User Datasets")
    print("=" * 60)

    users_dir = Path(args.users_dir)
    if not users_dir.exists():
        print(f"\nNo users directory found: {users_dir}")
        return

    users = [d for d in users_dir.iterdir() if d.is_dir()]

    if not users:
        print(f"\nNo user datasets found in {users_dir}")
        return

    print(f"\nFound {len(users)} user dataset(s):\n")

    for user_dir in sorted(users):
        user_id = user_dir.name

        # Check if processed
        lines_dir = user_dir / 'lines'
        gt_file = user_dir / 'ground_truth.json'

        status = "✓ Ready" if lines_dir.exists() and gt_file.exists() else "✗ Incomplete"

        # Count lines if available
        line_count = 0
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                line_count = len(json.load(f))

        print(f"  {user_id:20s} | {status:12s} | {line_count:3d} lines")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DisgraPhi Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process IAM database
  python ml/data/data.py download-iam --hf-token YOUR_HF_TOKEN

  # Generate practice packet
  python ml/data/data.py generate-packet --user-id user001 --output packet.pdf

  # Process photographed packet
  python ml/data/data.py process-packet --user-id user001 --images page1.jpg page2.jpg --ground-truth packet_ground_truth.json

  # Validate datasets
  python ml/data/data.py validate --mode iam --data-dir ml/data/raw/iam/processed
  python ml/data/data.py validate --mode personalization --user-dir ml/data/users/user001

  # List user datasets
  python ml/data/data.py list-users
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download IAM
    parser_iam = subparsers.add_parser(
        'download-iam',
        help='Download and process IAM Handwriting Database'
    )
    parser_iam.add_argument(
        '--data-dir',
        type=str,
        default='ml/data/raw/iam',
        help='Directory to store IAM data'
    )
    parser_iam.add_argument(
        '--hf-token',
        type=str,
        help='Hugging Face API token (or set HF_TOKEN env var)'
    )
    parser_iam.add_argument(
        '--repo-id',
        type=str,
        help='HF dataset repository ID (default: velocitatem/handwritten-baseline)'
    )
    parser_iam.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    parser_iam.set_defaults(func=download_iam)

    # Generate packet
    parser_gen = subparsers.add_parser(
        'generate-packet',
        help='Generate personalization practice packet'
    )
    parser_gen.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='Unique user identifier'
    )
    parser_gen.add_argument(
        '--output',
        type=str,
        default='packet.pdf',
        help='Output PDF path'
    )
    parser_gen.add_argument(
        '--ground-truth',
        type=str,
        help='Ground truth JSON output path (auto-generated if not specified)'
    )
    parser_gen.set_defaults(func=generate_packet)

    # Process packet
    parser_proc = subparsers.add_parser(
        'process-packet',
        help='Process photographed personalization packet'
    )
    parser_proc.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='User identifier'
    )
    parser_proc.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Paths to photographed packet pages'
    )
    parser_proc.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Path to ground truth JSON file'
    )
    parser_proc.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for processed data (default: ml/data/users/{user_id})'
    )
    parser_proc.set_defaults(func=process_packet)

    # Validate dataset
    parser_val = subparsers.add_parser(
        'validate',
        help='Validate dataset and show statistics'
    )
    parser_val.add_argument(
        '--mode',
        type=str,
        choices=['iam', 'personalization'],
        required=True,
        help='Dataset type to validate'
    )
    parser_val.add_argument(
        '--data-dir',
        type=str,
        help='IAM dataset directory (for --mode iam)'
    )
    parser_val.add_argument(
        '--user-dir',
        type=str,
        help='User dataset directory (for --mode personalization)'
    )
    parser_val.add_argument(
        '--show-sample',
        action='store_true',
        help='Display sample data'
    )
    parser_val.set_defaults(func=validate_dataset)

    # List users
    parser_list = subparsers.add_parser(
        'list-users',
        help='List available user datasets'
    )
    parser_list.add_argument(
        '--users-dir',
        type=str,
        default='ml/data/users',
        help='Users directory path'
    )
    parser_list.set_defaults(func=list_users)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        if os.getenv('DEBUG'):
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
