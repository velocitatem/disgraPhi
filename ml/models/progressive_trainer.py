"""
Progressive Training System

Enables users to see model improvements as they add samples, rather than
waiting for full dataset completion.

Key features:
- Checkpoint-based training (15, 30, 50, 75, 100 lines)
- Incremental model updates
- User feedback on model quality
- Adaptive sample recommendations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from alveslib import get_logger
from ml.models.batch_trainer import BatchLoRATrainer, TrainingJob

logger = get_logger("progressive-trainer")


@dataclass
class TrainingCheckpoint:
    """Represents a training checkpoint milestone."""
    name: str
    min_samples: int
    description: str
    expected_quality: str


# Standard checkpoints for progressive training
STANDARD_CHECKPOINTS = [
    TrainingCheckpoint(
        name="quick_start",
        min_samples=15,
        description="Initial personalization - rough accuracy",
        expected_quality="60-70% accurate (useful for simple text)"
    ),
    TrainingCheckpoint(
        name="basic",
        min_samples=30,
        description="Basic personalization - usable quality",
        expected_quality="75-85% accurate (good for most use cases)"
    ),
    TrainingCheckpoint(
        name="standard",
        min_samples=50,
        description="Standard personalization - recommended minimum",
        expected_quality="85-90% accurate (production ready)"
    ),
    TrainingCheckpoint(
        name="advanced",
        min_samples=75,
        description="Advanced personalization - high quality",
        expected_quality="90-95% accurate (excellent quality)"
    ),
    TrainingCheckpoint(
        name="optimal",
        min_samples=100,
        description="Optimal personalization - maximum quality",
        expected_quality="95%+ accurate (best possible quality)"
    ),
]


@dataclass
class UserProgress:
    """Tracks user's training progress."""
    user_id: str
    total_samples: int
    checkpoints_completed: List[str]
    latest_checkpoint: Optional[str]
    latest_cer: Optional[float]
    latest_model_path: Optional[str]
    last_updated: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProgress':
        """Create from dictionary."""
        return cls(**data)


class ProgressiveTrainer:
    """
    Manages progressive training for users.
    
    Automatically trains new model versions as users reach sample milestones,
    providing faster feedback and better engagement.
    
    Args:
        bootstrap_adapter: Path to bootstrap adapter
        users_base_dir: Base directory for user data
        models_base_dir: Base directory for storing trained models
        checkpoints: List of training checkpoints (default: STANDARD_CHECKPOINTS)
        auto_train: Whether to automatically train when checkpoints are reached
    """
    
    def __init__(
        self,
        bootstrap_adapter: str,
        users_base_dir: str = "ml/data/users",
        models_base_dir: str = "ml/models/user_loras",
        checkpoints: Optional[List[TrainingCheckpoint]] = None,
        auto_train: bool = True
    ):
        self.bootstrap_adapter = bootstrap_adapter
        self.users_base_dir = Path(users_base_dir)
        self.models_base_dir = Path(models_base_dir)
        self.checkpoints = checkpoints or STANDARD_CHECKPOINTS
        self.auto_train = auto_train
        
        # Progress tracking
        self.progress_dir = Path("ml/models/progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch trainer for executing training jobs
        self.batch_trainer = BatchLoRATrainer(
            bootstrap_adapter=bootstrap_adapter,
            output_base_dir=str(models_base_dir)
        )
        
        logger.info(f"Progressive trainer initialized with {len(self.checkpoints)} checkpoints")
    
    def get_user_progress(self, user_id: str) -> UserProgress:
        """
        Get current progress for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProgress object
        """
        progress_file = self.progress_dir / f"{user_id}.json"
        
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return UserProgress.from_dict(data)
        
        # Initialize new progress
        return UserProgress(
            user_id=user_id,
            total_samples=0,
            checkpoints_completed=[],
            latest_checkpoint=None,
            latest_cer=None,
            latest_model_path=None,
            last_updated=datetime.now().timestamp()
        )
    
    def save_user_progress(self, progress: UserProgress):
        """Save user progress to disk."""
        progress_file = self.progress_dir / f"{progress.user_id}.json"
        
        with open(progress_file, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2)
    
    def count_user_samples(self, user_id: str) -> int:
        """
        Count number of samples available for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of samples
        """
        user_dir = self.users_base_dir / user_id
        gt_file = user_dir / 'ground_truth.json'
        
        if not gt_file.exists():
            return 0
        
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
            return len(ground_truth)
    
    def get_next_checkpoint(self, current_samples: int) -> Optional[TrainingCheckpoint]:
        """
        Get the next checkpoint to reach.
        
        Args:
            current_samples: Number of samples user currently has
            
        Returns:
            Next checkpoint or None if all completed
        """
        for checkpoint in self.checkpoints:
            if current_samples < checkpoint.min_samples:
                return checkpoint
        
        return None
    
    def get_due_checkpoints(
        self,
        user_id: str,
        force: bool = False
    ) -> List[TrainingCheckpoint]:
        """
        Get checkpoints that should be trained for a user.
        
        Args:
            user_id: User identifier
            force: Force retraining of all checkpoints
            
        Returns:
            List of checkpoints to train
        """
        progress = self.get_user_progress(user_id)
        current_samples = self.count_user_samples(user_id)
        
        # Update sample count
        progress.total_samples = current_samples
        
        due_checkpoints = []
        
        for checkpoint in self.checkpoints:
            # Check if user has enough samples
            if current_samples < checkpoint.min_samples:
                continue
            
            # Check if already trained (unless forcing)
            if not force and checkpoint.name in progress.checkpoints_completed:
                continue
            
            due_checkpoints.append(checkpoint)
        
        return due_checkpoints
    
    def train_checkpoint(
        self,
        user_id: str,
        checkpoint: TrainingCheckpoint,
        num_samples: Optional[int] = None
    ) -> Dict:
        """
        Train a model at a specific checkpoint.
        
        Args:
            user_id: User identifier
            checkpoint: Checkpoint to train
            num_samples: Number of samples to use (default: checkpoint minimum)
            
        Returns:
            Training results
        """
        num_samples = num_samples or checkpoint.min_samples
        
        logger.info(
            f"Training {user_id} at checkpoint '{checkpoint.name}' "
            f"with {num_samples} samples"
        )
        
        user_dir = self.users_base_dir / user_id
        
        # Create training job
        job = TrainingJob(
            user_id=f"{user_id}_{checkpoint.name}",
            user_dir=str(user_dir),
            bootstrap_adapter=self.bootstrap_adapter,
            priority='normal'
        )
        
        # Train using batch trainer (single job)
        results = self.batch_trainer.train_batch([job], show_progress=True)
        
        result = results[job.user_id]
        
        if 'error' in result:
            logger.error(f"Training failed: {result['error']}")
            raise Exception(f"Training failed: {result['error']}")
        
        # Update progress
        progress = self.get_user_progress(user_id)
        
        if checkpoint.name not in progress.checkpoints_completed:
            progress.checkpoints_completed.append(checkpoint.name)
        
        progress.latest_checkpoint = checkpoint.name
        progress.latest_cer = result.get('cer')
        progress.latest_model_path = result.get('model_path')
        progress.last_updated = datetime.now().timestamp()
        
        self.save_user_progress(progress)
        
        logger.info(
            f"✓ Checkpoint '{checkpoint.name}' complete: "
            f"CER={result.get('cer', 0):.2f}%"
        )
        
        return result
    
    def process_user(
        self,
        user_id: str,
        force: bool = False
    ) -> List[Dict]:
        """
        Process all due checkpoints for a user.
        
        Args:
            user_id: User identifier
            force: Force retraining of all checkpoints
            
        Returns:
            List of training results
        """
        due_checkpoints = self.get_due_checkpoints(user_id, force=force)
        
        if not due_checkpoints:
            logger.info(f"No due checkpoints for {user_id}")
            return []
        
        logger.info(
            f"Processing {len(due_checkpoints)} checkpoints for {user_id}: "
            f"{[c.name for c in due_checkpoints]}"
        )
        
        results = []
        
        for checkpoint in due_checkpoints:
            try:
                result = self.train_checkpoint(user_id, checkpoint)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to train checkpoint '{checkpoint.name}': {e}")
                results.append({'checkpoint': checkpoint.name, 'error': str(e)})
        
        return results
    
    def get_recommendations(self, user_id: str) -> Dict:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Recommendations dictionary
        """
        progress = self.get_user_progress(user_id)
        current_samples = self.count_user_samples(user_id)
        next_checkpoint = self.get_next_checkpoint(current_samples)
        
        recommendations = {
            'user_id': user_id,
            'current_samples': current_samples,
            'latest_checkpoint': progress.latest_checkpoint,
            'latest_cer': progress.latest_cer,
            'next_checkpoint': None,
            'samples_needed': 0,
            'estimated_quality': None,
            'suggestions': []
        }
        
        if next_checkpoint:
            samples_needed = next_checkpoint.min_samples - current_samples
            
            recommendations['next_checkpoint'] = next_checkpoint.name
            recommendations['samples_needed'] = samples_needed
            recommendations['estimated_quality'] = next_checkpoint.expected_quality
            
            # Generate suggestions
            if samples_needed > 0:
                recommendations['suggestions'].append(
                    f"Add {samples_needed} more samples to unlock '{next_checkpoint.name}' "
                    f"checkpoint ({next_checkpoint.description})"
                )
                
                # Time estimate (30 seconds per sample)
                minutes = (samples_needed * 30) // 60
                recommendations['suggestions'].append(
                    f"Estimated time: ~{minutes} minutes"
                )
        else:
            # All checkpoints completed
            recommendations['suggestions'].append(
                "All checkpoints completed! Your model is at optimal quality."
            )
            
            # Encourage continuous learning
            if current_samples < 150:
                recommendations['suggestions'].append(
                    "Add more diverse samples to further improve accuracy on edge cases."
                )
        
        # Quality-based suggestions
        if progress.latest_cer and progress.latest_cer > 10:
            recommendations['suggestions'].append(
                f"Current accuracy: {100 - progress.latest_cer:.1f}%. "
                "Consider adding samples with challenging words or symbols."
            )
        
        return recommendations
    
    def generate_report(self, user_id: str) -> str:
        """
        Generate a human-readable progress report.
        
        Args:
            user_id: User identifier
            
        Returns:
            Formatted report string
        """
        progress = self.get_user_progress(user_id)
        recommendations = self.get_recommendations(user_id)
        
        report = []
        report.append("=" * 60)
        report.append(f"DisgraPhi Progress Report: {user_id}")
        report.append("=" * 60)
        report.append("")
        
        # Current status
        report.append(f"Total samples: {recommendations['current_samples']}")
        
        if progress.latest_checkpoint:
            report.append(f"Latest checkpoint: {progress.latest_checkpoint}")
            
            if progress.latest_cer is not None:
                accuracy = 100 - progress.latest_cer
                report.append(f"Current accuracy: {accuracy:.1f}%")
        
        report.append("")
        
        # Checkpoint progress
        report.append("Checkpoint Progress:")
        for checkpoint in self.checkpoints:
            status = "✓" if checkpoint.name in progress.checkpoints_completed else " "
            report.append(
                f"  [{status}] {checkpoint.name:15s} "
                f"({checkpoint.min_samples:3d} samples) - {checkpoint.description}"
            )
        
        report.append("")
        
        # Recommendations
        if recommendations['suggestions']:
            report.append("Recommendations:")
            for suggestion in recommendations['suggestions']:
                report.append(f"  • {suggestion}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """CLI for progressive training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Progressive training for DisgraPhi users"
    )
    
    parser.add_argument(
        '--bootstrap-adapter',
        type=str,
        required=True,
        help='Path to bootstrap adapter'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        help='User ID to process (default: all users)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['train', 'status', 'report', 'recommend'],
        default='train',
        help='Action to perform'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining of all checkpoints'
    )
    parser.add_argument(
        '--users-dir',
        type=str,
        default='ml/data/users',
        help='Base directory for user data'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ProgressiveTrainer(
        bootstrap_adapter=args.bootstrap_adapter,
        users_base_dir=args.users_dir
    )
    
    # Get user IDs
    if args.user_id:
        user_ids = [args.user_id]
    else:
        users_dir = Path(args.users_dir)
        user_ids = [d.name for d in users_dir.iterdir() if d.is_dir()]
    
    # Execute action
    if args.action == 'train':
        # Train all due checkpoints
        for user_id in user_ids:
            logger.info(f"\nProcessing {user_id}...")
            results = trainer.process_user(user_id, force=args.force)
            
            if results:
                logger.info(f"Completed {len(results)} checkpoints")
    
    elif args.action == 'status':
        # Show progress for all users
        for user_id in user_ids:
            progress = trainer.get_user_progress(user_id)
            current_samples = trainer.count_user_samples(user_id)
            
            print(f"\n{user_id}:")
            print(f"  Samples: {current_samples}")
            print(f"  Checkpoints: {len(progress.checkpoints_completed)}/{len(trainer.checkpoints)}")
            
            if progress.latest_checkpoint:
                print(f"  Latest: {progress.latest_checkpoint}")
                
                if progress.latest_cer is not None:
                    print(f"  Accuracy: {100 - progress.latest_cer:.1f}%")
    
    elif args.action == 'report':
        # Generate detailed reports
        for user_id in user_ids:
            report = trainer.generate_report(user_id)
            print(report)
    
    elif args.action == 'recommend':
        # Get recommendations
        for user_id in user_ids:
            recommendations = trainer.get_recommendations(user_id)
            
            print(f"\nRecommendations for {user_id}:")
            for suggestion in recommendations['suggestions']:
                print(f"  • {suggestion}")


if __name__ == '__main__':
    main()
