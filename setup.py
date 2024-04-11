import os
import platform
from distutils.core import setup

from setuptools import find_packages


# Usage
# if timeout add this argument --default-timeout=100
# in development: pip install --upgrade -e {path to Maghz or . if in dir}
# in prod: pip install {path to Maghz or . if in dir}


def is_cuda_available():
    # NVIDIA Docker images set NVIDIA_VISIBLE_DEVICES environment variable
    print('------------------')
    print(os.getenv('NVIDIA_VISIBLE_DEVICES', None))
    if os.getenv('NVIDIA_VISIBLE_DEVICES', None) in ['cpu', 'None', 'mps']:
        return False
    else:
        if os.getenv('NVIDIA_VISIBLE_DEVICES', None) is not None:
            return True

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
        []
elif is_linux:
    torch_deps = \
        []
elif is_mac:
    tf_deps = ['tensorflow-macos']

cuda_deps = []
if is_cuda_available():
    cuda_deps = ['flash-attn', 'vllm']
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
                         'absl-py==2.1.0',
                         'accelerate==0.29.2',
                         'aiohttp==3.9.3',
                         'aiosignal==1.3.1',
                         'altair==5.3.0',
                         'array2gif==1.0.4',
                         'astunparse==1.6.3',
                         'async-timeout==4.0.3',
                         'attrs==23.2.0',
                         'av==12.0.0',
                         'beautifulsoup4==4.12.3',
                         'bitsandbytes==0.41.3.post2',
                         'blis==0.7.11',
                         'bs4==0.0.2',
                         'catalogue==2.0.10',
                         'certifi==2024.2.2',
                         'chardet==5.2.0',
                         'charset-normalizer==3.3.2',
                         'click==8.1.7',
                         'coloredlogs==15.0.1',
                         'confection==0.1.4',
                         'cymem==2.0.8',
                         'datasets==2.16.1',
                         'dill==0.3.7',
                         'evaluate==0.4.1',
                         'exceptiongroup==1.2.0',
                         'filelock==3.13.4',
                         'flatbuffers==24.3.25',
                         'frozenlist==1.4.1',
                         'fsspec[http]==2023.10.0',
                         'fvcore==0.1.5.post20221221',
                         'gast==0.5.4',
                         'nltk',
                         'google-pasta==0.2.0',
                         'gputil==1.4.0',
                         'grpcio==1.62.1',
                         'h5py==3.11.0',
                         'huggingface-hub==0.22.2',
                         'humanfriendly==10.0',
                         'idna==3.6',
                         'imageio==2.34.0',
                         'importlib-metadata==7.1.0',
                         'iniconfig==2.0.0',
                         'iopath==0.1.10',
                         'jinja2==3.1.3',
                         'joblib==1.4.0',
                         'jsonschema==4.21.1',
                         'jsonschema-specifications==2023.12.1',
                         'keras==3.2.1',
                         'langcodes==3.3.0',
                         'lazy-loader==0.4',
                         'libclang==18.1.1',
                         'lightning-utilities==0.11.2',
                         'markdown==3.6',
                         'markdown-it-py==3.0.0',
                         'markupsafe==2.1.5',
                         'mdurl==0.1.2',
                         'ml-dtypes==0.3.2',
                         'mpmath==1.3.0',
                         'multidict==6.0.5',
                         'multiprocess==0.70.15',
                         'murmurhash==1.0.10',
                         'namex==0.0.7',
                         'networkx==3.2.1',
                         'numpy==1.24.4',
                         'opencv-python==4.9.0.80',
                         'opt-einsum==3.3.0',
                         'optimum==1.18.1',
                         'optree==0.11.0',
                         'packaging==24.0',
                         'pandas==2.2.2',
                         'parameterized==0.9.0',
                         'pathlib-abc==0.1.1',
                         'pathy==0.11.0',
                         'peft==0.10.0',
                         'pillow==10.3.0',
                         'pluggy==1.4.0',
                         'portalocker==2.8.2',
                         'preshed==3.0.9',
                         'protobuf==4.25.3',
                         'psutil==5.9.8',
                         'pyarrow==11.0.0',
                         'pyarrow-hotfix==0.6',
                         'pydantic==1.10.15',
                         'pygments==2.17.2',
                         'pytest==8.1.1',
                         'python-dateutil==2.9.0.post0',
                         'pytorchvideo==0.1.5',
                         'pytz==2024.1',
                         'pyyaml==6.0.1',
                         'referencing==0.34.0',
                         'regex==2023.12.25',
                         'requests==2.31.0',
                         'responses==0.18.0',
                         'rich==13.7.1',
                         'rpds-py==0.18.0',
                         'safetensors==0.4.2',
                         'scikit-image==0.22.0',
                         'scikit-learn==1.4.2',
                         'scipy==1.9.1',
                         'sentencepiece==0.2.0',
                         'six==1.16.0',
                         'smart-open==6.4.0',
                         'soupsieve==2.5',
                         'spacy==3.5.3',
                         'spacy-legacy==3.0.12',
                         'spacy-loggers==1.0.5',
                         'srsly==2.4.8',
                         'sympy==1.12',
                         'tabulate==0.9.0',
                         'tensorboard==2.16.2',
                         'tensorboard-data-server==0.7.2',
                         'tensorboardx==2.6.2.2',
                         'tensorflow==2.16.1',
                         'tensorflow-io-gcs-filesystem==0.36.0',
                         'termcolor==2.4.0',
                         'thinc==8.1.12',
                         'threadpoolctl==3.4.0',
                         'tifffile==2024.2.12',
                         'tokenizers==0.15.2',
                         'tomli==2.0.1',
                         'toolz==0.12.1',
                         'torch==2.1.2',
                         'torchaudio==2.1.2',
                         'torchdata==0.7.1',
                         'torchmetrics==1.3.2',
                         'torchtext',
                         'torchtyping==0.1.4',
                         'torchvision==0.16.2',
                         'tqdm==4.66.2',
                         'transformers[sentencepiece]==4.37.2',
                         'typeguard==4.2.0',
                         'typer==0.7.0',
                         'typing-extensions==4.8.0',
                         'tzdata==2024.1',
                         'urllib3==2.2.1',
                         'wasabi==1.1.2',
                         'werkzeug==3.0.2',
                         'wheel==0.43.0',
                         'wrapt==1.16.0',
                         'xxhash==3.4.1',
                         'yacs==0.1.8',
                         'yarl==1.9.4',
                         'zipp==3.18.1',
                     ] + cuda_deps,
    classifiers=[])
