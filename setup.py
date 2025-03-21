from setuptools import setup, find_packages

setup(
    name = 'PyTomo',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
    'aotools',
    'numpy<2.0.0'
    ],
)
