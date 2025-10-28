from setuptools import setup, find_packages

setup(
    name="disgraphi",
    version="0.1.0",
    packages=find_packages(include=['alveslib', 'alveslib.*', 'ml', 'ml.*']),
    install_requires=[
        "python-logging-loki",
        "python-dotenv",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "pillow",
        "numpy",
        "tqdm",
    ],
)
