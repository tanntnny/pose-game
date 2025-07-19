
from setuptools import setup, find_packages

setup(
    name='pose-estimation-game',
    version='0.1.0',
    description='A pose estimation game using MediaPipe and PyTorch',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'mediapipe',
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)
