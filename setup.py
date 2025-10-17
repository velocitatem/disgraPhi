from setuptools import setup, find_packages

setup(
    name="alveslib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-logging-loki",
        "python-dotenv",
    ],
)
