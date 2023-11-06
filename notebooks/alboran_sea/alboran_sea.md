# Alboran sea

```python
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr

from jaxparrow.tools import compute_coriolis_factor, compute_derivative, compute_spatial_step
from jaxparrow import cyclogeostrophy, geostrophy
```

```python
# utility functions

def dist(true: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    estimate[np.abs(true) < .1] = 0
    true[np.abs(true) < .1] = 0
    return true - estimate


def compute_norm_vorticity(u: np.ndarray, v: np.ndarray, dy_u: np.ndarray, dx_v: np.ndarray, 
                           mask: np.ndarray, f: np.ndarray) -> np.ma.masked_array:
    du_dy = compute_derivative(u, dy_u, axis=0)
    dv_dx = compute_derivative(v, dx_v, axis=1)
    return ma.masked_array(dv_dx - du_dy, mask) / f
```

## Input data

In this example, we use NEMO model outputs (SSH and velocities), stored in several netCDF files.
Data can be downloaded [here](https://1drv.ms/f/s!Aq7KsFIdmDGepjMT6o77ko-JRRZu?e=hpxeKa), and the files stored inside the `data` folder.

Measurements are located on a C-grid.


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


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_14_0.png?raw=true)
    


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
ax1.set_title("normalized vorticity - (NEMO - geos)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticity - (NEMO - geos) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_geos), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_geos), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```




    <matplotlib.colorbar.Colorbar at 0x147915af0>




    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_19_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_19_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_19_3.png?raw=true)
    


## Cyclogeostrophic balance

### Variational method

Cyclogeostrophic velocities are computed via the `cyclogeostrophy` function, using geostrophic velocities (here, the ones we previously computed), spatial steps, and the coriolis factors.


```python
u_var, v_var = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v)
```

    100%|██████████| 2000/2000 [00:03<00:00, 528.08it/s]



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
ax1.set_title("normalized vorticity - (NEMO - var)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticity - (NEMO - var) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_var), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_var), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```




    <matplotlib.colorbar.Colorbar at 0x141d70c10>




    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_25_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_25_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_25_3.png?raw=true)
    


### Penven method

We use the same function, but with the argument `method="iterative"`. 


```python
u_penven, v_penven = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, method="iterative")
```

    100%|██████████| 100/100 [00:00<00:00, 431.61it/s]



```python
u_penven = ma.masked_array(u_penven, mask_u)
v_penven = ma.masked_array(v_penven, mask_v)
```

#### Comparison to NEMO's velocities


```python
norm_vorticity_penven = compute_norm_vorticity(u_penven, v_penven, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - penven")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_penven, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - penven - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_penven, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - (NEMO - penven)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticity - (NEMO - penven) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_penven), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_penven), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```




    <matplotlib.colorbar.Colorbar at 0x16a0ac220>




    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_30_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_30_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_30_3.png?raw=true)
    


### Ioannou method

We use the same function, but with the arguments `method="iterative"`, and `use_res_filter=True`. 


```python
u_ioannou, v_ioannou = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, method="iterative", use_res_filter=True)
```

    100%|██████████| 100/100 [00:00<00:00, 343.00it/s]



```python
u_ioannou = ma.masked_array(u_ioannou, mask_u)
v_ioannou = ma.masked_array(v_ioannou, mask_v)
```

#### Comparison to NEMO's currents


```python
norm_vorticity_ioannou = compute_norm_vorticity(u_ioannou, v_ioannou, dy_u, dx_v, mask_ssh, coriolis_factor)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - ioannou")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_ioannou, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - NEMO - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - ioannou - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_ioannou, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - (NEMO - ioannou)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticity - (NEMO - ioannou) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_ioannou), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity, norm_vorticity_ioannou), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```




    <matplotlib.colorbar.Colorbar at 0x16a7a5fa0>




    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_35_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_35_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_35_3.png?raw=true)
    


#### Comparison to Penven velocities


```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - penven")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - ioannou")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity_penven, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_ioannou, cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - penven - clipped")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
ax2.set_title("normalized vorticity - ioannou - clipped")
ax2.set_xlabel("longitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity_penven, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, norm_vorticity_ioannou, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax2)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.set_title("normalized vorticity - (penven - ioannou)")
ax1.set_xlabel("longitude")
ax2.set_xlabel("longitude")
ax2.set_title("normalized vorticity - (penven - ioannou) - clipped")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, dist(norm_vorticity_penven, norm_vorticity_ioannou), cmap="RdBu_r", shading="auto",
                    norm=colors.CenteredNorm())
plt.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lon, lat, dist(norm_vorticity_penven, norm_vorticity_ioannou), 
                    cmap="RdBu_r", shading="auto", norm=colors.CenteredNorm(halfrange=halfrange))
plt.colorbar(im, ax=ax2)
```




    <matplotlib.colorbar.Colorbar at 0x295ae7ee0>




    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_37_1.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_37_2.png?raw=true)
    



    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_37_3.png?raw=true)
    

