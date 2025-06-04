# setup.py
from setuptools import setup, find_packages

setup(
    name="fingerprint-matching-pipeline",
    version="1.0.0",
    description="End-to-end fingerprint matching pipeline",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "fingerprint-match=main:main",
        ],
    },
)
