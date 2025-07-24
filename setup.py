import io
import os
import re
import sys

import setuptools

ROOT_DIR = os.path.dirname(__file__)
SUPPORTED_PYTHON_VERSIONS = [(3, 8), (3, 9), (3, 10), (3, 11), (3, 12)]

if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHON_VERSIONS:
    msg = (
        f'Detected Python version {".".join(map(str, sys.version_info[:2]))}, which is not supported. '
        f'Only Python {", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHON_VERSIONS)} are supported.'
    )
    raise RuntimeError(msg)


def parse_readme(readme: str) -> str:
    """Parse the README.md file to be pypi compatible."""
    # Replace the footnotes.
    readme = readme.replace('<!-- Footnote -->', '#')
    footnote_re = re.compile(r'\[\^([0-9]+)\]')
    readme = footnote_re.sub(r'<sup>[\1]</sup>', readme)

    # Remove the dark mode switcher
    mode_re = re.compile(
        r'<picture>[\n ]*<source media=.*>[\n ]*<img(.*)>[\n ]*</picture>',
        re.MULTILINE,
    )
    readme = mode_re.sub(r'<img\1>', readme)
    return readme


long_description = ''
readme_filepath = 'README.md'
if os.path.exists(readme_filepath):
    long_description = io.open(readme_filepath, 'r', encoding='utf-8').read()
    long_description = parse_readme(long_description)

with open('requirements/eval.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='internnav',
    version='0.0.1',
    packages=setuptools.find_packages(),
    author='OpenRobotLab',
    author_email='OpenRobotLab@pjlab.org.cn',
    license='Apache 2.0',
    readme='README.md',
    description='InternNav: A benchmark evaluation framework for navigation tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8, <=3.12',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    include_package_data=True,
)
