from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
        name='rl_spaces',
        version='0.1.0',
        packages=find_packages(),
        license='All Rights Reserved',
        author='Ceyer Wakilpoor',
        author_email='c.waki@gmail.com',
        description='Experimentation platform for farland rl algorithms.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://gitlab.mitre.org/autonomous-cyberdefense/acs/rl-platform',
        classifiers=[
                'Programming Language :: Python :: 3',
                'Operating System :: OS Independent'
        ],
        python_requires='>=3.6',
        install_requires=[
                'gym',
        ],
        entry_points={
                'console_scripts': [
                        'scraping = pacer_scraping.__main__:cli'
                        # todo ask about this
                ]
        }
)
