from setuptools import setup, find_packages

setup(
    name = 'PyTomo',
    version = '0.0.2',
    license = 'MIT',
    author = 'SpaceODT',
    packages = ['PyTomo'],
    url='https://github.com/cmcorreia/PyTomo',
    install_requires = [
    'aotools',
    'numpy',
    'astropy'
    ],
)
