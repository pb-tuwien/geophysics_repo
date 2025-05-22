## Setup

This package was designed for the use with [anaconda](https://www.anaconda.com/).

Open the terminal and navigate to the directory with the package (change to YOUR path):
```bash
cd path/to/the/directory
```
Now you create a new conda environment:
```bash
conda env create -f general_gp.yml
```
and activate it with:
```bash
conda activate general_gp
```
You can also use an existing environment. But please make sure that a `numpy` version between `1.26.4` and `2.0` is installed.  

When your environment is activated, run the following command to install the package:
```bash
pip install .
```
You could also install it in editable mode, which allows for modifications on the go:
```bash
pip install -e .
```

## Third-Party Software

This project uses the following third-party libraries:

- **TEMIP_sensitivity** (Apache 2.0): https://github.com/aignerlukas/TEMIP_sensitivity

By using this software, you agree to the terms of these licenses.

## Dependencies

This project depends on the following third-party libraries:

- [ResIPy](https://gitlab.com/hkex/resipy) (Licensed under GNU GPLv3)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD 3-Clause)
- [PyGIMLi](https://github.com/gimli-org/gimli) (Licensed under Apache 2.0)
- [empymod](https://github.com/emsig/empymod) (Licensed under Apache 2.0)
- [EMagPy](https://gitlab.com/hkex/emagpy) (Licensed under GNU GPLv3)

These are runtime dependencies and are not redistributed with this package.