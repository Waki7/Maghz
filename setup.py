import os
from distutils.core import setup

from setuptools import find_packages

# Usage
# if timeout add this argument --default-timeout=100
# in development: pip install -e C:\Users\ceyer\OneDrive\Documents\Projects\Maghz
# in prod: pip install C:\Users\ceyer\OneDrive\Documents\Projects\Maghz

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
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
    python_requires='>3.9',

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
        'torch',
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
        # pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
        # pip3 install torchdata==0.5.1 torchtext==0.14.1 (use conda if you can)


        # python -m spacy download de_core_news_sm
        # python -m spacy download en_core_web_sm
    ],

    # https://pypi.org/classifiers/
    classifiers=[]
)
