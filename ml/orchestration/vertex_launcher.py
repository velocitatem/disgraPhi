"""
Vertex AI launcher for training jobs.

This module handles the submission and monitoring of training jobs
to GCP Vertex AI, including hyperparameter tuning jobs.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .vertex_config import VertexAIConfig


class VertexAILauncher:
    """
    Launches and manages training jobs on GCP Vertex AI.
    
    This class provides a non-invasive way to run the existing train.py
    on Vertex AI without modifying the core training code.
    """
    
    def __init__(self, config: VertexAIConfig):
        """
        Initialize the launcher.
        
        Args:
            config: Vertex AI configuration
        """
        self.config = config
        self._check_prerequisites()
    
    def _check_prerequisites(self):
        """Check if required tools are available."""
        try:
            subprocess.run(
                ["gcloud", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "gcloud CLI not found. Please install it from: "
                "https://cloud.google.com/sdk/docs/install"
            )
    
    def package_training_code(self, output_dir: Optional[str] = None) -> str:
        """
        Package the training code as a Python package for Vertex AI.
        
        Args:
            output_dir: Directory to store the package (temp dir if None)
            
        Returns:
            Path to the packaged training code (.tar.gz)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find project root (where setup.py is)
        project_root = Path(__file__).parent.parent.parent
        
        # Create distribution package
        dist_cmd = [
            "python", "setup.py", "sdist",
            "--dist-dir", str(output_path)
        ]
        
        print(f"Packaging training code from {project_root}...")
        result = subprocess.run(
            dist_cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to package training code:\n{result.stderr}"
            )
        
        # Find the created package
        packages = list(output_path.glob("*.tar.gz"))
        if not packages:
            raise RuntimeError(f"No package found in {output_path}")
        
        package_path = packages[0]
        print(f"Created package: {package_path}")
        
        return str(package_path)
    
    def upload_to_gcs(self, local_path: str, gcs_path: str) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (gs://...)
            
        Returns:
            GCS path of uploaded file
        """
        print(f"Uploading {local_path} to {gcs_path}...")
        
        result = subprocess.run(
            ["gsutil", "cp", local_path, gcs_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload to GCS:\n{result.stderr}"
            )
        
        print(f"Upload complete: {gcs_path}")
        return gcs_path
    
    def create_training_job_spec(self, package_gcs_uri: str) -> Dict[str, Any]:
        """
        Create the training job specification for Vertex AI.
        
        Args:
            package_gcs_uri: GCS URI of the training package
            
        Returns:
            Job specification dictionary
        """
        worker_pool = self.config.get_worker_pool_spec()
        
        # Add package URI to the container spec
        worker_pool["containerSpec"]["pythonPackageSpec"]["packageUris"] = [
            package_gcs_uri
        ]
        
        # Add GCS paths for data and output if specified
        if self.config.data_dir_gcs:
            worker_pool["containerSpec"]["pythonPackageSpec"]["args"].extend([
                "--data_dir", self.config.data_dir_gcs
            ])
        
        if self.config.output_dir_gcs:
            worker_pool["containerSpec"]["pythonPackageSpec"]["args"].extend([
                "--output_dir", self.config.output_dir_gcs
            ])
        
        job_spec = {
            "displayName": self.config.display_name,
            "jobSpec": {
                "workerPoolSpecs": [worker_pool],
                "serviceAccount": self.config.service_account,
                "network": self.config.network,
                "timeout": f"{self.config.timeout_seconds}s",
            }
        }
        
        # Add hyperparameter tuning config if enabled
        hp_config = self.config.get_hyperparameter_tuning_config()
        if hp_config:
            job_spec["studySpec"] = hp_config["studySpec"]
            job_spec["maxTrialCount"] = hp_config["maxTrialCount"]
            job_spec["parallelTrialCount"] = hp_config["parallelTrialCount"]
        
        return job_spec
    
    def submit_job(self, dry_run: bool = False) -> Optional[str]:
        """
        Submit a training job to Vertex AI.
        
        Args:
            dry_run: If True, only print the job spec without submitting
            
        Returns:
            Job ID if submitted, None if dry_run
        """
        # Package training code
        package_path = self.package_training_code()
        
        # Upload to GCS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = Path(package_path).name
        gcs_package_path = f"{self.config.staging_bucket}/packages/{timestamp}/{package_name}"
        
        if not dry_run:
            package_gcs_uri = self.upload_to_gcs(package_path, gcs_package_path)
        else:
            package_gcs_uri = gcs_package_path
        
        # Create job specification
        job_spec = self.create_training_job_spec(package_gcs_uri)
        
        if dry_run:
            print("\n" + "="*80)
            print("DRY RUN - Job specification:")
            print("="*80)
            print(json.dumps(job_spec, indent=2))
            print("="*80)
            return None
        
        # Write job spec to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(job_spec, f, indent=2)
            job_spec_path = f.name
        
        try:
            # Submit job using gcloud
            cmd = [
                "gcloud", "ai", "custom-jobs", "create",
                f"--region={self.config.region}",
                f"--display-name={self.config.display_name}",
                f"--config={job_spec_path}",
                f"--project={self.config.project_id}",
                "--format=json"
            ]
            
            print(f"\nSubmitting training job to Vertex AI...")
            print(f"Project: {self.config.project_id}")
            print(f"Region: {self.config.region}")
            print(f"Display name: {self.config.display_name}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            job_info = json.loads(result.stdout)
            job_id = job_info.get("name", "").split("/")[-1]
            
            print(f"\n✓ Job submitted successfully!")
            print(f"Job ID: {job_id}")
            print(f"View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job_id}?project={self.config.project_id}")
            
            return job_id
            
        finally:
            # Clean up temp file
            Path(job_spec_path).unlink(missing_ok=True)
    
    def monitor_job(self, job_id: str, poll_interval: int = 60):
        """
        Monitor a running training job.
        
        Args:
            job_id: The job ID to monitor
            poll_interval: Seconds between status checks
        """
        import time
        
        print(f"\nMonitoring job {job_id}...")
        print("Press Ctrl+C to stop monitoring (job will continue running)")
        
        try:
            while True:
                result = subprocess.run(
                    [
                        "gcloud", "ai", "custom-jobs", "describe",
                        job_id,
                        f"--region={self.config.region}",
                        f"--project={self.config.project_id}",
                        "--format=json"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                job_info = json.loads(result.stdout)
                state = job_info.get("state", "UNKNOWN")
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {state}")
                
                if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                    print(f"\nJob finished with state: {state}")
                    if state == "FAILED":
                        error = job_info.get("error", {})
                        print(f"Error: {error}")
                    break
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\nStopped monitoring (job is still running)")
    
    def cancel_job(self, job_id: str):
        """
        Cancel a running training job.
        
        Args:
            job_id: The job ID to cancel
        """
        print(f"Cancelling job {job_id}...")
        
        subprocess.run(
            [
                "gcloud", "ai", "custom-jobs", "cancel",
                job_id,
                f"--region={self.config.region}",
                f"--project={self.config.project_id}"
            ],
            check=True
        )
        
        print("✓ Job cancelled")
