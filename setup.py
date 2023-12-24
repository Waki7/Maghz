import os
import platform
from distutils.core import setup

from setuptools import find_packages


# Usage
# if timeout add this argument --default-timeout=100
# in development: pip install --upgrade -e {path to Maghz or . if in dir}
# in prod: pip install {path to Maghz or . if in dir}


def is_cuda_available():
    # Check CUDA_HOME environment variable
    if os.getenv('CUDA_HOME', None) is None:
        return False

    # Check if nvcc is available
    import subprocess
    try:
        subprocess.check_output(['nvcc', '--version'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
os_name = platform.system()
is_windows = os_name == 'Windows'
is_linux = os_name == 'Linux'
is_mac = os_name == 'Darwin'

torch_deps = []
tf_deps = []
if is_windows:
    torch_deps = \
        [
            "torch@https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp38-cp38-win_amd64.whl"
            "torchvision@https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp38-cp38-win_amd64.whl"]
elif is_linux:
    torch_deps = \
        [
            "torch@https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp38-cp38-linux_x86_64.whl"
            "torchvision@https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp38-cp38-linux_x86_64.whl"]
elif is_mac:
    tf_deps = ['tensorflow-macos']

cuda_deps = []
if is_cuda_available():
    cuda_deps = ['flash-attn', 'auto-gptq>=0.4.2']
    tf_deps = ['tensorflow-gpu==2.12.0']

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
    packages=find_packages() + ["mgz", "spaces"],
    package_dir={
        "mgz": "./mgz",
        "spaces": "./spaces",
    },
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
        'spacy==3.5.3',
        'torchvision',
        'scipy==1.9.1',
        'numpy==1.24.3',
        'tensorflow>=2.11.0',
        'torchvision',
        'transformers==4.35.2',
        'opencv-python',
        'array2gif',
        'Pillow',
        'pyyaml',
        'scikit-image',
        'pytest',
        'torchdata',
        'datasets',
        'GPUtil',
        'altair',
        'evaluate',
        'chardet',
        'pytorchvideo',
        'scikit-learn',
        'accelerate',
        'bs4',
        'typing_extensions==4.4.0',
        'peft',
        'bitsandbytes==0.41.0',
        'optimum',
        'torch>=2.1.0',
        'torchtext',
        'pyarrow==11.0.0',
        # 'torchvision',
        # 'torchaudio',
        # 'torchdata',
        # 'torchtext',
        # 'protobuf>=3.20',
        # 'inspect-it',
    ],  # + cuda_deps + tf_deps,
    classifiers=[]
)
