#!/usr/bin/env python3
"""
DisgraPhi Scaling Demo

This script demonstrates the new batch training and progressive training features.

Run this after you have:
1. Downloaded and processed the IAM database
2. Trained a bootstrap model
3. Generated and processed user packets

Usage:
    python examples/scaling_demo.py --bootstrap-adapter ml/models/bootstrap/best_model
"""

import argparse
from pathlib import Path


def demo_progressive_training(bootstrap_adapter: str, user_id: str = "demo_user"):
    """
    Demonstrate progressive training with checkpoints.
    
    Shows how users get feedback at 15, 30, 50 sample milestones.
    """
    print("\n" + "=" * 60)
    print("PROGRESSIVE TRAINING DEMO")
    print("=" * 60)
    
    try:
        from ml.models.progressive_trainer import ProgressiveTrainer
        
        trainer = ProgressiveTrainer(
            bootstrap_adapter=bootstrap_adapter,
            users_base_dir="ml/data/users",
            models_base_dir="ml/models/user_loras"
        )
        
        # Get user progress
        progress = trainer.get_user_progress(user_id)
        print(f"\nUser: {user_id}")
        print(f"Total samples: {progress.total_samples}")
        print(f"Checkpoints completed: {len(progress.checkpoints_completed)}")
        
        # Get recommendations
        recommendations = trainer.get_recommendations(user_id)
        print(f"\nRecommendations:")
        for suggestion in recommendations['suggestions']:
            print(f"  • {suggestion}")
        
        # Generate full report
        report = trainer.generate_report(user_id)
        print(f"\n{report}")
        
    except ImportError as e:
        print(f"⚠ Cannot run demo: {e}")
        print("Install dependencies: pip install -r requirements.txt")


def demo_batch_training(bootstrap_adapter: str):
    """
    Demonstrate batch training of multiple users.
    
    Shows how to train 8 users in parallel on a single GPU.
    """
    print("\n" + "=" * 60)
    print("BATCH TRAINING DEMO")
    print("=" * 60)
    
    try:
        from ml.models.batch_trainer import BatchLoRATrainer, TrainingJob
        
        # Find users to train
        users_dir = Path("ml/data/users")
        if not users_dir.exists():
            print(f"⚠ Users directory not found: {users_dir}")
            print("Create user datasets first with: python ml/data/data.py process-packet")
            return
        
        user_dirs = [d for d in users_dir.iterdir() if d.is_dir()]
        
        if not user_dirs:
            print(f"⚠ No users found in {users_dir}")
            return
        
        print(f"\nFound {len(user_dirs)} users to train")
        
        # Create batch trainer
        trainer = BatchLoRATrainer(
            bootstrap_adapter=bootstrap_adapter,
            num_parallel=min(8, len(user_dirs)),  # Don't exceed user count
            output_base_dir="ml/models/user_loras"
        )
        
        # Create jobs (limit to first 3 for demo)
        jobs = [
            TrainingJob(
                user_id=d.name,
                user_dir=str(d),
                bootstrap_adapter=bootstrap_adapter
            )
            for d in user_dirs[:3]
        ]
        
        print(f"\nTraining {len(jobs)} users in parallel...")
        print("(Limited to 3 for demo - remove limit in production)")
        
        # Train batch
        results = trainer.train_batch(jobs)
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        for user_id, result in results.items():
            if 'error' in result:
                print(f"\n✗ {user_id}: Failed - {result['error']}")
            else:
                print(f"\n✓ {user_id}:")
                print(f"  CER: {result['cer']:.2f}%")
                print(f"  Training time: {result['training_time']:.1f}s")
                print(f"  Samples: {result['num_samples']}")
                print(f"  Model path: {result['model_path']}")
        
        # Calculate averages
        successful = [r for r in results.values() if 'error' not in r]
        if successful:
            avg_cer = sum(r['cer'] for r in successful) / len(successful)
            avg_time = sum(r['training_time'] for r in successful) / len(successful)
            total_time = sum(r['training_time'] for r in successful)
            
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Successful: {len(successful)}/{len(results)}")
            print(f"Average CER: {avg_cer:.2f}%")
            print(f"Average time per user: {avg_time:.1f}s")
            print(f"Total training time: {total_time:.1f}s")
            print(f"Throughput: {len(successful) / (total_time / 3600):.1f} users/hour")
        
    except ImportError as e:
        print(f"⚠ Cannot run demo: {e}")
        print("Install dependencies: pip install -r requirements.txt")


def demo_queue_mode(bootstrap_adapter: str):
    """
    Demonstrate training queue for production deployment.
    """
    print("\n" + "=" * 60)
    print("TRAINING QUEUE DEMO")
    print("=" * 60)
    
    try:
        from ml.models.batch_trainer import BatchLoRATrainer, TrainingQueueManager
        
        print("\nSetting up training queue...")
        
        trainer = BatchLoRATrainer(
            bootstrap_adapter=bootstrap_adapter,
            num_parallel=8
        )
        
        queue = TrainingQueueManager(
            batch_trainer=trainer,
            batch_size=8,
            batch_timeout=60
        )
        
        print("✓ Queue manager initialized")
        print("\nQueue configuration:")
        print(f"  Batch size: 8 users")
        print(f"  Batch timeout: 60 seconds")
        print(f"  Max queue size: 1000 jobs")
        
        # Simulate job submission
        print("\nTo submit jobs in production:")
        print("  queue.submit_job(user_id='alice', user_dir='ml/data/users/alice', priority='high')")
        
        print("\nTo start queue processor:")
        print("  python ml/models/batch_trainer.py --queue-mode --bootstrap-adapter <path>")
        
        # Show queue status
        status = queue.get_queue_length()
        print(f"\nCurrent queue status:")
        print(f"  High priority: {status['high_priority']}")
        print(f"  Normal priority: {status['normal_priority']}")
        print(f"  Total: {status['total']}")
        
    except ImportError as e:
        print(f"⚠ Cannot run demo: {e}")
        print("Install dependencies: pip install -r requirements.txt")


def demo_configuration():
    """Show how to configure scaling settings."""
    print("\n" + "=" * 60)
    print("CONFIGURATION DEMO")
    print("=" * 60)
    
    import yaml
    
    config_path = "ml/config/scaling.yml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\nScaling configuration loaded from: {config_path}")
        
        # Batch training settings
        batch_config = config.get('batch_training', {})
        print(f"\nBatch Training:")
        print(f"  Parallel jobs: {batch_config.get('num_parallel')}")
        print(f"  Batch timeout: {batch_config.get('batch_timeout')}s")
        print(f"  Max queue size: {batch_config.get('max_queue_size')}")
        
        # Progressive training settings
        prog_config = config.get('progressive_training', {})
        print(f"\nProgressive Training:")
        print(f"  Enabled: {prog_config.get('enabled')}")
        print(f"  Checkpoints: {len(prog_config.get('checkpoints', []))}")
        
        for checkpoint in prog_config.get('checkpoints', [])[:3]:
            print(f"    • {checkpoint['name']}: {checkpoint['min_samples']} samples")
        
        # Features
        features = config.get('features', {})
        enabled_features = [k for k, v in features.items() if v]
        print(f"\nEnabled Features:")
        for feature in enabled_features:
            print(f"  ✓ {feature}")
        
        print(f"\nTo modify settings, edit: {config_path}")
        
    except Exception as e:
        print(f"⚠ Error loading config: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DisgraPhi Scaling Features Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all demos
  python examples/scaling_demo.py --bootstrap-adapter ml/models/bootstrap/best_model
  
  # Show specific demo
  python examples/scaling_demo.py --demo batch --bootstrap-adapter ml/models/bootstrap/best_model
  
  # Show progressive training for specific user
  python examples/scaling_demo.py --demo progressive --user-id alice --bootstrap-adapter ml/models/bootstrap/best_model
        """
    )
    
    parser.add_argument(
        '--bootstrap-adapter',
        type=str,
        help='Path to bootstrap adapter (required for training demos)'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['all', 'progressive', 'batch', 'queue', 'config'],
        default='all',
        help='Which demo to run'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        default='demo_user',
        help='User ID for progressive training demo'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DisgraPhi Scaling Features Demo")
    print("=" * 60)
    print("\nThis demo shows the new batch training and progressive training features.")
    print("See docs/SCALING_GUIDE.md for full documentation.\n")
    
    # Run requested demos
    if args.demo in ['all', 'config']:
        demo_configuration()
    
    if args.demo in ['all', 'progressive']:
        if not args.bootstrap_adapter:
            print("\n⚠ --bootstrap-adapter required for progressive training demo")
        else:
            demo_progressive_training(args.bootstrap_adapter, args.user_id)
    
    if args.demo in ['all', 'batch']:
        if not args.bootstrap_adapter:
            print("\n⚠ --bootstrap-adapter required for batch training demo")
        else:
            demo_batch_training(args.bootstrap_adapter)
    
    if args.demo in ['all', 'queue']:
        if not args.bootstrap_adapter:
            print("\n⚠ --bootstrap-adapter required for queue demo")
        else:
            demo_queue_mode(args.bootstrap_adapter)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read docs/SCALING_GUIDE.md for detailed usage")
    print("  2. Read docs/DATA_COLLECTION_RESEARCH.md for improvement roadmap")
    print("  3. Configure ml/config/scaling.yml for your deployment")
    print("  4. Start scaling DisgraPhi to thousands of users!")


if __name__ == '__main__':
    main()
