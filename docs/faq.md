# FAQ

## How can I use `jaxparrow` to estimate cyclogeostrophic currents from large models outputs?

`Dask` is probably your best bet to handle large datasets that do not fit into (CPU or GPU) memory.

On a CPU backend, one can use a chunk size of 1 along the time dimension and map the call to [`cyclogeostrophy`](api.md#jaxparrow.cyclogeostrophy.cyclogeostrophy) onto the dataset:

```python
import dask.array as da
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jaxparrow.cyclogeostrophy import cyclogeostrophy


def do_one_block(in_block):
    ucg, vcg, ug, vg = cyclogeostrophy(
        jnp.asarray(in_block.ssh.values), jnp.asarray(in_block.lat.values), jnp.asarray(in_block.lon.values), 
        return_geos=True, return_grids=False
    )

    out_block = xr.Dataset(
        {
            "ucg": (in_block.ssh.dims, np.asarray(ucg)[None, :, :]),
            "vcg": (in_block.ssh.dims, np.asarray(vcg)[None, :, :]),
            "ug": (in_block.ssh.dims, np.asarray(ug)[None, :, :]),
            "vg": (in_block.ssh.dims, np.asarray(vg)[None, :, :]),
        }
    )

    return out_block


nt = ds.time.size
ny = ds.lat.size
nx = ds.lon.size

empty_arr = da.empty((nt, ny, nx), chunks=(1, ny, nx), dtype=np.float32)
template = xr.Dataset(
    {
        "ucg": (ds.ssh.dims, empty_arr),
        "vcg": (ds.ssh.dims, empty_arr),
        "ug": (ds.ssh.dims, empty_arr),
        "vg": (ds.ssh.dims, empty_arr),
    },
)

result = xr.map_blocks(do_one_block, ds, template=template)
result = result.assign_coords({
    "time": ds.time,
    "lat": (("y", "x"), ds.lat),
    "lon": (("y", "x"), ds.lon),
})

result.to_zarr(OUT_PATH, compute=True, consolidated=False)
```

On a GPU backend, the previous approach can be combined with `jax.vmap` to further speed up computations by processing multiple time slices in parallel on the GPU:

```python
import dask.array as da
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jaxparrow.cyclogeostrophy import cyclogeostrophy


vmap_cyclogeostrophy = jax.vmap(
    lambda *args: cyclogeostrophy(*args, return_geos=True, return_grids=False),
    in_axes=(0, None, None)
)


def do_one_block_vmap(in_block: xr.Dataset):
    ucg_3d, vcg_3d, ug_3d, vg_3d = vmap_cyclogeostrophy(
        jnp.asarray(in_block.ssh.values), jnp.asarray(in_block.lat.values), jnp.asarray(in_block.lon.values)
    )

    out_block = xr.Dataset(
        {
            "ucg": (in_block.ssh.dims, np.asarray(ucg_3d)),
            "vcg": (in_block.ssh.dims, np.asarray(vcg_3d)),
            "ug": (in_block.ssh.dims, np.asarray(ug_3d)),
            "vg": (in_block.ssh.dims, np.asarray(vg_3d)),
        }
    )

    return out_block


with dask.config.set(scheduler="synchronous"):
    nt = ds.time.size
    ny = ds.lat.size
    nx = ds.lon.size

    empty_arr = da.empty((nt, ny, nx), chunks=(BATCH_SIZE, ny, nx), dtype=np.float32)
    template = xr.Dataset(
        {
            "ucg": (ds.ssh.dims, empty_arr),
            "vcg": (ds.ssh.dims, empty_arr),
            "ug": (ds.ssh.dims, empty_arr),
            "vg": (ds.ssh.dims, empty_arr),
        },
    )

    result = xr.map_blocks(do_one_block_vmap, ds, template=template)
    result = result.assign_coords({
        "time": ds.time,
        "lat": (("y", "x"), ds.lat),
        "lon": (("y", "x"), ds.lon),
    })

    result.to_zarr(OUT_PATH, compute=True, consolidated=False)
```

Note that in this case, you will need to force `Dask` to use the synchronous scheduler as `JAX` is not multi-threaded.

## I am getting very large current velocity estimates

From our experience, this can happen when the input SSH data contains unbalanced signals.
To mitigate this, we clip updates during the gradient descent minimization.
This is achieved using the [`optax.clip`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip) transformation:

```python
import optax

optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=5e-3)
)
```

And then pass the `optimizer` object as the `optim` argument of the [`cyclogeostrophy`](api.md#jaxparrow.cyclogeostrophy.cyclogeostrophy) function.

We also recommend using `JAX` floating point types with sufficient precision, e.g., `float64`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```
