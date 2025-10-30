from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="disgraphi",
    version="0.1.0",
    author="velocitatem",
    description="Personalized handwriting recognition for everyone",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/velocitatem/disgraPhi",
    packages=find_packages(include=["ml", "ml.*", "alveslib", "alveslib.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core ML dependencies
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        
        # Image processing
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        
        # Utilities
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "huggingface_hub>=0.19.0",
        
        # Logging
        "python-logging-loki>=0.3.1",
        "python-dotenv>=1.0.0",
        
        # QR code processing
        "pyzbar>=0.1.9",
        
        # Web scraping (for data collection)
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "inference": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "webapp": [
            "streamlit>=1.28.0",
            "flask>=3.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.8.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "streamlit>=1.28.0",
            "flask>=3.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "disgraphi-train=ml.models.train:main",
            "disgraphi-data=ml.data.data:main",
            "disgraphi-inference=ml.inference:main",
        ],
    },
)
