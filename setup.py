import io
import os
from setuptools import setup, find_packages


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    # >>> read("project_name", "VERSION")
    '0.1.0'
    # >>> read("README.md")
    # ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


setup(
    name = 'tomoAO',
    version=open("tomoAO/version.py").read().split('=')[1].strip().strip('"'),
    license = 'MIT',
    author = 'SpaceODT',
    description='Python package with atmospheric tomography generic functions and tools',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    packages = find_packages(),
    url='https://github.com/cmcorreia/PyTomo',
    install_requires = [
    'aotools',
    'numpy',
    'astropy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
    ]
)
