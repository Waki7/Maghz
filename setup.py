import os
from distutils.core import setup

from setuptools import find_packages

# Usage
# if timeout add this argument --default-timeout=100
# in development: pip install -e {path to Maghz or . if in dir}
# in prod: pip install {path to Maghz or . if in dir}

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
operating_system = os.name
is_windows = operating_system == 'nt'
is_linux = operating_system == 'posix'
is_mac = operating_system == 'mac'

torch_deps = []
if is_windows:
    torch_deps = \
        [
            "torch@https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp38-cp38-win_amd64.whl"
            "torchvision@https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp38-cp38-win_amd64.whl"]
else:
    torch_deps = \
        [
            "torch@https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp38-cp38-linux_x86_64.whl"
            "torchvision@https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp38-cp38-linux_x86_64.whl"]

try:
    with open(os.path.join(current_directory, 'README.md'),
              encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    # Name of the package
    name='Maghz',

    # Packages to include into the distribution
    packages=find_packages(),

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='1.0.0',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='',

    # Short description of your library
    description='',

    # Long description of your library
    long_description=long_description,
    long_description_context_type='text/markdown',

    # Your name
    author='Ceyer Wakilpoor',

    # Your email
    author_email='c.waki7@gmail.com',

    # Either the link to your github or to your website
    url='',

    # Link from which the project can be downloaded
    download_url='',

    # List of keyword arguments
    keywords=[],
    python_requires='>3.8',

    # List of packages to install with this one
    install_requires=[
        'tensorboard',
        'tensorboardX',
        'torchtext',
        'spacy',
        'gym',
        'torchvision',
        'scipy',
        'numpy',
        'tensorflow',
        'torchvision',
        'transformers',
        'opencv-python',
        'array2gif',
        'Pillow',
        'stable-baselines',
        'pyyaml',
        'scikit-image',
        'pytest',
        'spacy==3.4',
        'torchtext',
        'torchdata',
        'datasets',
        'GPUtil',
        'spacy',
        'altair',
        'evaluate',
        'scikit-learn',
        'accelerate',
        'bs4',
        'typing_extensions==4.4.0',
        'peft',
        'bitsandbytes',
        'optimum',
        'torch>=2.1.0',
        'torchvision',
        'torchaudio',
        'torchdata',
        'torchtext',
        'tensorflow-gpu==2.8.0',
        'protobuf==3.20',
        'auto-gptq==0.4.2',
        'flash-attn',
        'inspect-it',
    ],
    classifiers=[]
)
