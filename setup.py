from setuptools import find_packages
from distutils.core import setup

setup(
    name='multiverse',
    author='Yoonho Shin',
    version='1.0',
    description='Accompanying code for Environment Curriculum Generation via Adversarial Environment Generation',
    python_requires='>=3.8',
    install_requires=[
        'numpy<1.24',
        'scipy>=0.13.0',
        'matplotlib',
        'openai',
        'opencv-python',
        'pydelatin',
        'pyfqmr',
        'hydra-core',
        'wandb',
        'gpustat',
        'tqdm',
        'ipdb',
    ],
    packages=find_packages()
)
