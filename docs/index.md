# tomoAO

Python package with atmospheric tomography generic functions and tools.

## Installation

1. Install OOPAO

    ```sh
    git clone https://github.com/cheritier/OOPAO
    pip install -e 'path to OOPAO'
    ```

2. Install cupy according to your CUDA version (optional — enables GPU
   covariance matrix computation)

    ```sh
    pip install cupy-cuda11x   # CUDA 11.x
    pip install cupy-cuda12x   # CUDA 12.x
    ```

3. Install tomoAO

    ```sh
    pip install tomoAO
    ```

    Or from source:

    ```sh
    git clone https://github.com/SpaceODT/tomoAO
    pip install -e 'path to tomoAO'
    ```

## API Reference

| Object | Purpose |
| --- | --- |
| [`AOSystem`](reference/aosystem.md) | Container for telescope, atmosphere, DM, guide stars and subaperture masks |
| [`tomoReconstructor`](reference/reconstructor.md) | Linear MMSE tomographic reconstructor built from an `AOSystem` |
