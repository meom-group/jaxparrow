```python
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import optax
import xarray as xr

from jaxparrow.tools import compute_coriolis_factor, compute_spatial_step
from jaxparrow import cyclogeostrophy, geostrophy

%reload_ext autoreload
%autoreload 2
```


```python
# utility functions

def dist(true: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    estimate[np.abs(true) < .1] = 0
    true[np.abs(true) < .1] = 0
    return true - estimate


from jaxparrow.tools.tools import compute_derivative
def compute_norm_vorticity(u: np.ndarray, v: np.ndarray, dy_u: np.ndarray, dx_v: np.ndarray, 
                           mask: np.ndarray, f: np.ndarray) -> np.ma.masked_array:
    du_dy = compute_derivative(u, dy_u, axis=0)
    dv_dx = compute_derivative(v, dx_v, axis=1)
    return ma.masked_array(dv_dx - du_dy, mask) / f
```

# Alboran sea

## Input data

In this example, we use NEMO model outputs (SSH and velocities), stored in several netCDF files.
Measurements are located on a C-grid.

Data can be downloaded [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/jaxparrow/alboransea.tar.gz), and the files extracted to the `data` folder.
The next cell does this for you, assuming wget and tar are available.


```python
!wget -P data https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/jaxparrow/alboransea.tar.gz
!tar -xzf data/alboransea.tar.gz -C data
!rm data/alboransea.tar.gz
```
    



```python
data_dir = "data"
name_mask = "mask_alboransea.nc"
name_coord = "coordinates_alboransea.nc"
name_ssh = "alboransea_sossheig.nc"
name_u = "alboransea_sozocrtx.nc"
name_v = "alboransea_somecrty.nc"
```


```python
ds_coord = xr.open_dataset(os.path.join(data_dir, name_coord))
lon = ds_coord.nav_lon.values
lat = ds_coord.nav_lat.values

ds_mask = xr.open_dataset(os.path.join(data_dir, name_mask))
mask_ssh = ds_mask.tmask[0,0].values
mask_u = ds_mask.umask[0,0].values
mask_v = ds_mask.vmask[0,0].values

ds_ssh = xr.open_dataset(os.path.join(data_dir, name_ssh))
lon_ssh = ds_ssh.nav_lon.values
lat_ssh = ds_ssh.nav_lat.values
ssh = ds_ssh.sossheig[0].values

ds_u = xr.open_dataset(os.path.join(data_dir, name_u))
lon_u = ds_u.nav_lon.values
lat_u = ds_u.nav_lat.values
uvel = ds_u.sozocrtx[0].values

ds_v = xr.open_dataset(os.path.join(data_dir, name_v))
lon_v = ds_v.nav_lon.values
lat_v = ds_v.nav_lat.values
vvel = ds_v.somecrty[0].values
```

We use `masked_array` to restrict the domain to the marine area.


```python
mask_u = 1 - mask_u
mask_v = 1 - mask_v
mask_ssh = 1 - mask_ssh
```


```python
uvel = ma.masked_array(uvel, mask_u)
vvel = ma.masked_array(vvel, mask_v)
ssh = ma.masked_array(ssh, mask_ssh)
```


```python
lon_u = ma.masked_array(lon_u, mask_u)
lat_u = ma.masked_array(lat_u, mask_u)
lon_v = ma.masked_array(lon_v, mask_v)
lat_v = ma.masked_array(lat_v, mask_v)
lon_ssh = ma.masked_array(lon_ssh, mask_ssh)
lat_ssh = ma.masked_array(lat_ssh, mask_ssh)
```

### Compute spatial steps

The netCDF files we use as input do not contain the spatial steps required to compute derivatives later.
The sub-module `tools` provides the utility function `compute_spatial_step` to compute them from our grids. It applies Von Neuman boundary conditions to those fields.


```python
dx_ssh, dy_ssh = compute_spatial_step(lat_ssh, lon_ssh)
dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
dx_v, dy_v = compute_spatial_step(lat_v, lon_v)
```

### Coriolis factor

Estimating the velocities also involve the Coriolis factor, which varies with the latitude.
The function `compute_coriolis_factor` from the sub-module `tools` might be used here.


```python
coriolis_factor = compute_coriolis_factor(lat)
coriolis_factor_u = compute_coriolis_factor(lat_u)
coriolis_factor_v = compute_coriolis_factor(lat_v)
```

### Visualising SSH and currents


```python
norm_vorticity = compute_norm_vorticity(uvel, vvel, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("SSH (m)")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, ssh, cmap="turbo", shading="auto")
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_15_0.png?raw=true)
    


## Geostrophic balance

We estimate the geostrophic velocities using the `geostrophy` function, given the SSH, the spatial steps, and the coriolis factors.


```python
u_geos, v_geos = geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
```


```python
u_geos = ma.masked_array(u_geos, mask_u)
v_geos = ma.masked_array(v_geos, mask_v)
```

### Comparison to NEMO's velocities


```python
vmin = -4
vmax = -vmin
halfrange = 2

norm_vorticity_geos = compute_norm_vorticity(u_geos, v_geos, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - geostrophy")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_geos, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - geostrophy - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_geos, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticities difference - (NEMO - geos)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticities difference - (NEMO - geos) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_geos), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_geos), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_20_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_20_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_20_3.png?raw=true)
    


## Cyclogeostrophic balance

### Variational method

Cyclogeostrophic velocities are computed via the `cyclogeostrophy` function, using geostrophic velocities (here, the ones we previously computed), spatial steps, and the coriolis factors.

The optimizer can be specified as a string (assuming it refers to an `optax` [common optimizers](https://optax.readthedocs.io/en/latest/api.html#)): `optim = "sgd"` for example.
Or designed using a more refined strategy:


```python
lr_scheduler = optax.exponential_decay(1e-2, 200, .5)
base_optim = optax.sgd(learning_rate=lr_scheduler)
optim = optax.chain(optax.clip(1), base_optim)
```


```python
u_var, v_var, losses = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, n_it=1000, optim=optim, return_losses=True)
```

    100%|██████████| 1000/1000 [00:01<00:00, 636.12it/s] 



```python
plt.plot(losses)
plt.title("cyclogeostrophic disequilibrium - variational")
plt.xlabel("steps")
plt.ylabel("disequilibrium")
plt.show()

```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_25_0.png?raw=true)
    



```python
u_var = ma.masked_array(u_var, mask_u)
v_var = ma.masked_array(v_var, mask_v)
```

#### Comparison to NEMO's velocities


```python
norm_vorticity_var = compute_norm_vorticity(u_var, v_var, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - variational")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_var, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - variational - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_var, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticities difference - (NEMO - var)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticities difference - (NEMO - var) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_var), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_var), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_28_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_28_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_28_3.png?raw=true)
    


### Iterative method

We use the same function, but with the argument `method="iterative"`. 


```python
u_iterative, v_iterative, losses = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, method="iterative", return_losses=True)
```

     20%|██        | 20/100 [00:00<00:00, 653.72it/s]



```python
plt.plot(losses)
plt.title("cyclogeostrophic disequilibrium - iterative")
plt.xlabel("steps")
plt.ylabel("disequilibrium")
plt.show()

```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_31_0.png?raw=true)
    



```python
u_iterative = ma.masked_array(u_iterative, mask_u)
v_iterative = ma.masked_array(v_iterative, mask_v)
```

#### Comparison to NEMO's velocities


```python
norm_vorticity_iterative = compute_norm_vorticity(u_iterative, v_iterative, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - iterative")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_iterative, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - iterative - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_iterative, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticities difference - (NEMO - iterative)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticities difference - (NEMO - iterative) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_iterative), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_iterative), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_34_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_34_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_34_3.png?raw=true)
    


### Iterative method, with filter

We use the same function, but with the arguments `method="iterative"`, and `use_res_filter=True`. 


```python
u_it_filter, v_it_filter, losses = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, method="iterative", use_res_filter=True, return_losses=True)
```

    100%|██████████| 100/100 [00:00<00:00, 546.62it/s]



```python
plt.plot(losses)
plt.title("cyclogeostrophic disequilibrium - it_filter")
plt.xlabel("steps")
plt.ylabel("disequilibrium")
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_37_0.png?raw=true)
    



```python
u_it_filter = ma.masked_array(u_it_filter, mask_u)
v_it_filter = ma.masked_array(v_it_filter, mask_v)
```

#### Comparison to NEMO's currents


```python
norm_vorticity_it_filter = compute_norm_vorticity(u_it_filter, v_it_filter, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - it_filter")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_it_filter, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - it_filter - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_it_filter, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticities difference - (NEMO - it_filter)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticities difference - (NEMO - it_filter) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_it_filter), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_it_filter), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_40_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_40_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_40_3.png?raw=true)
    

