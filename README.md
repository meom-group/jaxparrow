# jaxparrow

***jaxparrow*** implements a novel approach based on a variational formulation to compute the inversion of the cyclogeostrophic balance.

It leverages the power of [JAX](https://jax.readthedocs.io/en/latest/), to efficiently solve the inversion as an optimization problem. 
Given the Sea Surface Height (SSH) or the geostrophic velocity field of an ocean system, **jaxparrow** estimates the velocity field that best satisfies the cyclogeostrophic balance.

## Installation

The package is Pip-installable:
```shell
pip install jaxparrow
```

**<ins>However</ins>**, users with access to GPUs or TPUs should first install JAX separately in order to fully benefit from its high-performance computing capacities. 
See [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html). \
By default, **jaxparrow** will install a CPU-only version of JAX if no other version is already present in the Python environment.

## Usage

### As a package

Two functions are directly available from `jaxparrow`:

- `geostrophy`: computes the geostrophic velocity field (returns two `numpy 2darray`) from a SSH `2darray`, two `2darray` of spatial steps, and two `2darray` of Coriolis factors.
- `cyclogeostrophy`: computes the cyclogeostrophic velocity field (returns two `2darray`) from two `2darray` of geostrophic velocities, four `2darray` of spatial steps, and two `2darray` of Coriolis factors.

*Because **jaxparrow** uses [C-grids](https://xgcm.readthedocs.io/en/latest/grids.html) the velocity fields are represented on two grids, and the SSH on one grid.*

In a Python script, assuming that the input grids have already been initialised / imported, it would simply resort to:

```python
from jaxparrow import cyclogeostrophy, geostrophy

u_geos, v_geos = geostrophy(ssh=ssh,    
                            dx_ssh=dx_ssh, dy_ssh=dy_ssh,
                            coriolis_factor_u=coriolis_factor_u, coriolis_factor_v=coriolis_factor_v)
u_cyclo, v_cyclo = cyclogeostrophy(u_geos=u_geos, v_geos=v_geos,
                                   dx_u=dx_u, dx_v=dx_v, dy_u=dy_u, dy_v=dy_v,
                                   coriolis_factor_u=coriolis_factor_u, coriolis_factor_v=coriolis_factor_v)
```

By default, the `cyclogeostrophy` function relies on our variational method.
Its `method` argument provides the ability to use an iterative method instead, either the one described by [Penven *et al.*](https://doi.org/10.1016/j.dsr2.2013.10.015), or the one by [Ioannou *et al.*](https://doi.org/10.1029/2019JC015031).
Additional arguments also give a finer control over the three approaches hyperparameters. \
See [**jaxparrow** documentation](docs/_build/index.html) for more details.

[Notebooks](notebooks/README.md) are available as step-by-step examples.

### As an executable

***TBP***
