from setuptools import setup, find_packages

setup(
    name = 'PyTomo',
    version = '0.0.1',
    license = 'MIT',
    author = 'SpaceODT',
    packages = ['PyTomo'],
    url='https://github.com/cmcorreia/PyTomo',
    install_requires = [
    'aotools',
    'numpy<2.0.0'
    ],
)
