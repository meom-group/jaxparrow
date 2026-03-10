# jaxparrow

![Python](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/meom-group/jaxparrow/master/.github/workflows/python-package.yml&label=Python&query=$.jobs.build.strategy.matrix["python-version"])
[![PyPi](https://img.shields.io/badge/dynamic/xml?url=https://pypi.org/rss/project/jaxparrow/releases.xml&label=PyPi&query=/rss/channel/item[1]/title)](https://pypi.org/project/jaxparrow/)
![Tests](https://github.com/meom-group/jaxparrow/actions/workflows/python-package.yml/badge.svg)
[![Docs](https://app.readthedocs.org/projects/jaxparrow/badge/)](https://jaxparrow.readthedocs.io/)
[![DOI](https://zenodo.org/badge/702998298.svg)](https://zenodo.org/badge/latestdoi/702998298)

`jaxparrow` implements a novel approach based on a minimization-based formulation to compute the inversion of the cyclogeostrophic balance.

It leverages the power of [`JAX`](https://jax.readthedocs.io/en/latest/), to efficiently solve the inversion as a minimization problem.
Given the Sea Surface Height (SSH) field of an ocean system, `jaxparrow` estimates the velocity field that best satisfies the cyclogeostrophic balance.

A comprehensive documenation is available: [https://jaxparrow.readthedocs.io/en/latest/](https://jaxparrow.readthedocs.io/en/latest/)!

## Installation

`jaxparrow` is Pip-installable:

```shell
pip install jaxparrow
```

**<ins>However</ins>**, users with access to GPUs or TPUs should first install `JAX` separately in order to fully benefit from its high-performance computing capacities.
See [JAX instructions](https://jax.readthedocs.io/en/latest/installation.html).
By default, `jaxparrow` will install a CPU-only version of JAX if no other version is already present in the Python environment.

## Usage

Estimating the cyclogeostrophic currents from a given Sea Surface Height field can be achieved using any of the following methods:

- [`minimization_based`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.minimization_based),
- [`gradient_wind`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.gradient_wind),
- [`fixed_point`](https://jaxparrow.readthedocs.io/en/latest/api/#jaxparrow.cyclogeostrophy.fixed_point).

They all need at least:

- a SSH field (a `2d jax.Array`),
- the latitude and longitude grids at the T points (two `2d jax.Array`).

In a Python script, assuming that the input grids have already been initialised / imported, estimating the cyclogeostrophic currents for a single timestamp would resort to:

```python
from jaxparrow import minimization_based

mb_result = minimization_based(ssh_2d, lat_2d, lon_2d)

ucg_2d = mb_result.ucg  # 2d jax.Array
vcg_2d = mb_result.vcg  # 2d jax.Array
```

*Because `jaxparrow` uses [C-grids](https://xgcm.readthedocs.io/en/latest/grids.html) the velocity fields are represented on two grids (U and V), and the tracer fields (such as SSH) on one grid (T).*
We provide functions computing some kinematics (such as velocities magnitude, normalized relative vorticity, or kinetic energy) accounting for these gridding system:

```python
from jaxparrow.tools.kinematics import magnitude

uv_cg = magnitude(ucg, vcg)
```

To vectorise the estimation of the cyclogeostrophy along a first time dimension, one aims to use `jax.vmap`.

```python
import jax

vmap_cyclogeostrophy = jax.vmap(cyclogeostrophy, in_axes=(0, None, None))
mb_result = vmap_cyclogeostrophy(ssh_3d, lat_2d, lon_2d)

ucg_3d = mb_result.ucg  # 3d jax.Array
vcg_3d = mb_result.vcg  # 3d jax.Array
```

See `jaxparrow` [documentation](https://jaxparrow.readthedocs.io/en/latest/) for more details (including the API description and step-by-step examples).

## Contributing

Contributions are welcomed!
See [CONTRIBUTING.md](https://github.com/meom-group/jaxparrow/blob/main/CONTRIBUTING.md) and [CONDUCT.md](https://github.com/meom-group/jaxparrow/blob/main/CONDUCT.md) to get started.

## How to cite

If you use this software, please cite it: [CITATION.cff](https://github.com/meom-group/jaxparrow/blob/main/CITATION.cff).
Thank you!
