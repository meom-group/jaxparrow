# jaxparrow

![Python](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/meom-group/jaxparrow/master/.github/workflows/python-package.yml&label=Python&query=$.jobs.build.strategy.matrix["python-version"])
[![PyPi](https://img.shields.io/badge/dynamic/xml?url=https://pypi.org/rss/project/jaxparrow/releases.xml&label=PyPi&query=/rss/channel/item[1]/title)](https://pypi.org/project/jaxparrow/)
![Tests](https://github.com/meom-group/jaxparrow/actions/workflows/python-package.yml/badge.svg)
[![Docs](https://github.com/meom-group/jaxparrow/actions/workflows/python-documentation.yml/badge.svg)](https://jaxparrow.readthedocs.io/)
[![DOI](https://zenodo.org/badge/702998298.svg)](https://doi.org/10.5281/zenodo.13886071)

`jaxparrow` implements a novel approach based on a variational formulation to compute the inversion of the cyclogeostrophic balance.

It leverages the power of [`JAX`](https://jax.readthedocs.io/en/latest/), to efficiently solve the inversion as an optimization problem. 
Given the Sea Surface Height (SSH) field of an ocean system, **jaxparrow** estimates the velocity field that best satisfies the cyclogeostrophic balance.

See the full [documentation](https://jaxparrow.readthedocs.io/en/latest/)!

## Installation

`jaxparrow` is Pip-installable:
```shell
pip install jaxparrow
```

**<ins>However</ins>**, users with access to GPUs or TPUs should first install `JAX` separately in order to fully benefit from its high-performance computing capacities. 
See [JAX instructions](https://jax.readthedocs.io/en/latest/installation.html). \
By default, `jaxparrow` will install a CPU-only version of JAX if no other version is already present in the Python environment.

## Usage

### As a package

The function you are most probably looking for is `cyclogeostrophy`.
It computes the cyclogeostrophic velocity field (returned as two `2darray`) from:
- a SSH field (a `2darray`), 
- the latitude and longitude grids at the T points (two `2darray`).

In a Python script, assuming that the input grids have already been initialised / imported, estimating the cyclogeostrophic velocities for a single timestamp would resort to:

```python
from jaxparrow import cyclogeostrophy

u_cyclo_2d, v_cyclo_2d = cyclogeostrophy(ssh_2d, lat_2d, lon_2d)
```

*Because `jaxparrow` uses [C-grids](https://xgcm.readthedocs.io/en/latest/grids.html) the velocity fields are represented on two grids (U and V), and the tracer fields (such as SSH) on one grid (T).* \
We provide functions computing some kinematics (such as velocities magnitude, normalized relative vorticity, or kinematic energy) accounting for these gridding system:

```python
from jaxparrow.tools.kinematics import magnitude

uv_cyclo_2d, v_cyclo_2d = magnitude(u_cyclo_2d, v_cyclo_2d, interpolate=True)
```

To vectorise the estimation of the cyclogeostrophy across a first time dimension, one aims to use `jax.vmap`.

```python
import jax

vmap_cyclogeostrophy = jax.vmap(cyclogeostrophy, in_axes=(0, None, None))
u_cyclo_3d, v_cyclo_3d = vmap_cyclogeostrophy(ssh_3d, lat_2d, lon_2d)
```

By default, the `cyclogeostrophy` function relies on our variational method.
Its `method` argument provides the ability to use an iterative method instead, either the one described by [Penven *et al.*](https://doi.org/10.1016/j.dsr2.2013.10.015), or the one by [Ioannou *et al.*](https://doi.org/10.1029/2019JC015031).
Additional arguments also give a finer control over the three approaches hyperparameters. \
See `jaxparrow` [API documentation](https://jaxparrow.readthedocs.io/en/latest/api.html) for more details.

[Notebooks](https://jaxparrow.readthedocs.io/en/latest/examples.html) are available as step-by-step examples.

### As an executable

**jaxparrow** is also available from the command line:
```shell
jaxparrow --conf_path conf.yml
```
The YAML configuration file `conf.yml` instruct where input netCDF files are locally stored, and how to retrieve variables and coordinates from them.
It also provides the path of the output netCDF file. Optionally, it can specify which cyclogeostrophic approach should be applied and its hyperparameters.

An example configuration file detailing all the required and optional entries can be found [here](https://github.com/meom-group/jaxparrow/blob/main/docs/example-conf.yml).

## Contributing

Contributions of all sorts are welcomed!
See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONDUCT.md](CONDUCT.md) to get started.

## How to cite

If you use this software, please cite it as specified in [CITATION.cff](CITATION.cff).
Thank you!
