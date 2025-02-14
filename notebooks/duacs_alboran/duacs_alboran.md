# Hands-on `jaxparrow` with Duacs data

The aim of this notebook is to illustrate how `jaxparrow` can be employed to derive geostrophic **and cyclogeostrophic** currents on a C-grid from Sea Surface Height (SSH) observations.  
The demo focuses on the **Alboran Sea**, a highly energetic area of the Mediterranean Sea ([Ioannou et al. 2019](https://doi.org/10.1029/2019JC015031)).

We use the European Seas Gridded L4 Sea Surface Heights And Derived Variables Reprocessed dataset ([description](https://data.marine.copernicus.eu/product/SEALEVEL_EUR_PHY_L4_MY_008_068/description), [reference](https://doi.org/10.48670/moi-00141)).  
This product provides daily average of SSH, and geostrophic currents, on a rectilinear A-grid, with a spatial resolution of 1/16°.

We need to install some dependencies first:


```python
!pip install ipympl matplotlib cartopy
!pip install copernicusmarine jaxparrow jaxtyping numpy xarray

%reload_ext autoreload
%autoreload 2
```

## Accessing Duacs data

Copernicus Marine datasets can be accessed through the [Copernicus Marine Toolbox](https://help.marine.copernicus.eu/en/collections/4060068-copernicus-marine-toolbox) API.


```python
import copernicusmarine as cm
```

The [API](https://help.marine.copernicus.eu/en/collections/5821001-python-library-api) allows to download subsets of the datasets by restricting the spatial and temporal domains, and the variables.


```python
import pandas as pd

spatial_extent = (-5.3538, -1.1883, 35.0707, 36.8415)  # spatial extent (lon0, lon1, lat0, lat1) of the Alboran Sea
temporal_slice = (pd.to_datetime("2019-01-01T00:00:00"), pd.to_datetime("2019-12-31T23:59:59"))  # we look at the 2019 data for our demo
variables = ["adt", "ugos", "vgos"]  # we retrieve SSH and geostrophic currents (for comparison) data

dataset_options = {
    "dataset_id": "cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D",
    "variables": variables,
    "minimum_longitude": spatial_extent[0],
    "maximum_longitude": spatial_extent[1],
    "minimum_latitude": spatial_extent[2],
    "maximum_latitude": spatial_extent[3],
    "start_datetime": temporal_slice[0],
    "end_datetime": temporal_slice[1]
}
duacs_ds = cm.open_dataset(**dataset_options)
```

### Visualisation

Lets visualise how the SSH, and the magnitude of the geostrophic currents, evolve over the time period.


```python
import numpy as np

duacs_ds = duacs_ds.assign(uvgos=np.sqrt(duacs_ds.ugos ** 2 + duacs_ds.vgos ** 2))
```


```python
import matplotlib.pyplot as plt

from duacs_visualisation import AnimatedSSHCurrent
%matplotlib widget

anim = AnimatedSSHCurrent(duacs_ds, ("adt", "uvgos"), ("Duacs", "Duacs"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_9_0.png)


## Geostrophic currents using `jaxparrow`

`jaxparrow` uses C-grids, following NEMO convention. U, V, and F points are automatically derived from the T points.


```python
import jax.numpy as jnp  # we manipulate jax.Array

lat_t = jnp.ones((duacs_ds.latitude.size, duacs_ds.longitude.size)) * duacs_ds.latitude.data.reshape(-1, 1)
lon_t = jnp.ones((duacs_ds.latitude.size, duacs_ds.longitude.size)) * duacs_ds.longitude.data
```

The spatial domain covers sea and land, we derive tbe mask to exclude the land parts of the domain from the `adt` invalid values.


```python
adt_t = jnp.asarray(duacs_ds.adt.data)
mask = ~(jnp.isfinite(adt_t))
```

And we compute the geostrophic currents using the `geostrophy` function.

Rather than looping over our time indices, we can vectorise the `geostrophy` function over the time axis and compute the geostrophic currents at every time point using the vectorise version.


```python
import jax
import jaxparrow as jpw

vmap_geostrophy = jax.vmap(jpw.geostrophy, in_axes=(0, None, None, 0), out_axes=(0, 0, None, None, None, None))

ug_u, vg_v, lat_u, lon_u, lat_v, lon_v = vmap_geostrophy(adt_t, lat_t, lon_t, mask)
```

To visualise the results, we compute the magnitude of the velocity.


```python
from jaxparrow.tools.kinematics import magnitude

uvg_t = jax.vmap(magnitude, in_axes=(0, 0))(ug_u, vg_v)
```

We store everything in an `xarray` `Dataset`.


```python
import xarray as xr

gos_ds = xr.Dataset(
    {
        "adt": (["time", "latitude", "longitude"], adt_t),
        "ug": (["time", "latitude_u", "longitude_u"], ug_u),
        "vg": (["time", "latitude_v", "longitude_v"], vg_v),
        "uvg": (["time", "latitude", "longitude"], uvg_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude, 
        "latitude_u": np.unique(lat_u).astype(np.float32), "longitude_u": np.unique(lon_u).astype(np.float32), 
        "latitude_v": np.unique(lat_v).astype(np.float32), "longitude_v": np.unique(lon_v).astype(np.float32)
    }
)
```

### Visualisation


```python
anim = AnimatedSSHCurrent(gos_ds, ("adt", "uvg"), ("Duacs", "jaxparrow (geos)"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_22_0.png)


#### Geostrophic inter-comparison

For sanity check we can compare the two geostrophic reconstructions.


```python
from duacs_visualisation import AnimatedCurrents

gos_ds = xr.Dataset(
    {
        "uvg": (["time", "latitude", "longitude"], duacs_ds.uvgos.data),
        "uvg_jpw": (["time", "latitude", "longitude"], uvg_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude
    }
)

anim = AnimatedCurrents(gos_ds, ("uvg", "uvg_jpw"), ("Duacs", "jaxparrow"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_24_0.png)


## Cyclogeostrophic currents using `jaxparrow`

Now, lets see the results of the inversion methods of the cyclogeostrophic currents.

### Our variational approach


```python
vmap_cyclogeostrophy = jax.vmap(jpw.cyclogeostrophy, in_axes=(0, None, None, 0), out_axes=(0, 0, None, None, None, None))

uc_var_u, vc_var_v, lat_u, lon_u, lat_v, lon_v = vmap_cyclogeostrophy(adt_t, lat_t, lon_t, mask)

uvc_var_t = jax.vmap(magnitude, in_axes=(0, 0))(uc_var_u, vc_var_v)
```


```python
cgos_var_ds = xr.Dataset(
    {
        "adt": (["time", "latitude", "longitude"], adt_t),
        "uc": (["time", "latitude_u", "longitude_u"], uc_var_u),
        "vc": (["time", "latitude_v", "longitude_v"], vc_var_v),
        "uvc": (["time", "latitude", "longitude"], uvc_var_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude, 
        "latitude_u": np.unique(lat_u).astype(np.float32), "longitude_u": np.unique(lon_u).astype(np.float32), 
        "latitude_v": np.unique(lat_v).astype(np.float32), "longitude_v": np.unique(lon_v).astype(np.float32)
    }
)
```

### The iterative method


```python
vmap_cyclogeostrophy_it = jax.vmap(
    lambda *args: jpw.cyclogeostrophy(*args, method="iterative"), 
    in_axes=(0, None, None, 0), out_axes=(0, 0, None, None, None, None)
)

uc_it_u, vc_it_v, lat_u, lon_u, lat_v, lon_v = vmap_cyclogeostrophy_it(adt_t, lat_t, lon_t, mask)

uvc_it_t = jax.vmap(magnitude, in_axes=(0, 0))(uc_it_u, vc_it_v)
```


```python
cgos_it_ds = xr.Dataset(
    {
        "adt": (["time", "latitude", "longitude"], adt_t),
        "uc": (["time", "latitude_u", "longitude_u"], uc_it_u),
        "vc": (["time", "latitude_v", "longitude_v"], vc_it_v),
        "uvc": (["time", "latitude", "longitude"], uvc_it_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude, 
        "latitude_u": np.unique(lat_u).astype(np.float32), "longitude_u": np.unique(lon_u).astype(np.float32), 
        "latitude_v": np.unique(lat_v).astype(np.float32), "longitude_v": np.unique(lon_v).astype(np.float32)
    }
)
```

### Visualisation


```python
anim = AnimatedSSHCurrent(cgos_var_ds, ("adt", "uvc"), ("Duacs", "jaxparrow (cyclogeos)"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_33_0.png)


#### Comparison with geostrophy


```python
g_cg_ds = xr.Dataset(
    {
        "uvg": (["time", "latitude", "longitude"], uvg_t),
        "uvc": (["time", "latitude", "longitude"], uvc_var_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude
    }
)

anim = AnimatedCurrents(g_cg_ds, ("uvg", "uvc"), ("geostrophy", "cyclogeostrophy"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_35_0.png)


#### Variational and iterative methods comparison


```python
cg_ds = xr.Dataset(
    {
        "uv_it": (["time", "latitude", "longitude"], uvc_it_t),
        "uv_var": (["time", "latitude", "longitude"], uvc_var_t)
    },
    coords={
        "time": duacs_ds.time,
        "latitude": duacs_ds.latitude, "longitude": duacs_ds.longitude
    }
)

anim = AnimatedCurrents(cg_ds, ("uv_it", "uv_var"), ("iterative", "variational"))
plt.show()
```



![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/duacs_alboran/output_37_0.png)