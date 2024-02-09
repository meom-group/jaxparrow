```python
import os

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import optax
import xarray as xr

from jaxparrow.tools import tools
from jaxparrow import cyclogeostrophy, geostrophy
```

```python
# utility functions

vmin = -4
vmax = -vmin
dpi_ref = 100
full_width_px = 1600


def get_figsize(width_ratio, wh_ratio=1):
    fig_width = full_width_px / dpi_ref * width_ratio
    fig_height = fig_width / wh_ratio
    return fig_width, fig_height
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

We use `array` of masks to restrict the domain to the marine area.


```python
mask_u = 1 - mask_u
mask_v = 1 - mask_v
mask_ssh = 1 - mask_ssh
```

### Visualising SSH and currents


```python
# compute some characteristics
norm_vorticity = tools.compute_norm_vorticity(uvel, vvel, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v)
magnitude = ma.masked_array(tools.compute_magnitude(uvel, vvel), mask_ssh)

mmin = np.nanmin(magnitude)
mmax = np.nanmax(magnitude)

# interpolate to the center of the cells
norm_vorticity_t = tools.interpolate_south_west(norm_vorticity, axis=0, neuman=False)
norm_vorticity_t = tools.interpolate_south_west(norm_vorticity_t, axis=1, neuman=False)
uvel_t = tools.interpolate_south_west(uvel, axis=1, neuman=False)
vvel_t = tools.interpolate_south_west(vvel, axis=0, neuman=False)
```

```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 20/3), 
                                  subplot_kw={"projection": ccrs.PlateCarree()})

ax1.set_title("Sea Surface Height")
im = ax1.pcolormesh(lon, lat, ma.masked_array(ssh, mask_ssh), 
                    cmap="turbo", shading="auto",
                    transform=ccrs.PlateCarree())
clb1 = plt.colorbar(im, ax=ax1)
clb1.ax.set_title("SSH (m)")

ax2.set_title("Current velocity")
im = ax2.pcolormesh(lon, lat, magnitude, 
                    shading="auto",
                    transform=ccrs.PlateCarree())
ax2.quiver(lon[::5, ::5], lat[::5, ::5], 
           ma.masked_array(uvel_t, mask_u)[::5, ::5], ma.masked_array(vvel_t, mask_v)[::5, ::5], 
           color="k")
clb2 = plt.colorbar(im, ax=ax2)
clb2.ax.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$ (m/s)")

ax3.set_title("Current normalized vorticity")
im = ax3.pcolormesh(lon, lat, norm_vorticity_t, 
                    cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())
clb3 = plt.colorbar(im, ax=ax3)
clb3.ax.set_title("$\\xi / f$")

plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_10_0.png?raw=true)


## Geostrophic balance

We estimate the geostrophic velocities using the `geostrophy` function, given the SSH, the spatial steps, and the coriolis factors.


```python
u_geos, v_geos = geostrophy(ssh, lat_ssh, lon_ssh, lat_u, lat_v, mask_ssh, mask_u, mask_v)

norm_vorticity_geos = tools.compute_norm_vorticity(u_geos, v_geos, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v)
norm_vorticity_geos_t = tools.interpolate_south_west(norm_vorticity_geos, axis=0, neuman=False)
norm_vorticity_geos_t = tools.interpolate_south_west(norm_vorticity_geos_t, axis=1, neuman=False)
```

### Comparison to NEMO's velocities


```python
fig, axs = plt.subplots(2, 2, figsize=get_figsize(2/3, 12.66/6), 
                        subplot_kw={"projection": ccrs.PlateCarree()})

axs[0, 0].set_title("NEMO data")
_ = axs[0, 0].pcolormesh(lon, lat, norm_vorticity_t, 
                         cmap="RdBu_r", shading="auto", 
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

axs[0, 1].set_title("Geostrophy")
im1 = axs[0, 1].pcolormesh(lon, lat, norm_vorticity_geos_t, 
                           cmap="RdBu_r", shading="auto", 
                           vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())

_ = axs[1, 0].pcolormesh(lon, lat, magnitude, 
                         shading="auto", 
                         vmin=mmin, vmax=mmax,
                         transform=ccrs.PlateCarree())

im2 = axs[1, 1].pcolormesh(lon, lat, tools.compute_magnitude(u_geos, v_geos), 
                           shading="auto", 
                           vmin=mmin, vmax=mmax,
                           transform=ccrs.PlateCarree())

fig.tight_layout()
fig.subplots_adjust(right=0.89, wspace=0.01)

cbar_ax1 = fig.add_axes([0.9, 0.51, 0.01, 0.38])
_ = fig.colorbar(im1, cax=cbar_ax1)
cbar_ax1.set_title("$\\xi / f$")

cbar_ax2 = fig.add_axes([0.9, 0.05, 0.01, 0.38])
_ = fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$")

plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_14_0.png?raw=true)
    

## Cyclogeostrophic balance

### Variational method

Cyclogeostrophic velocities are computed via the `cyclogeostrophy` function, using geostrophic velocities (here, the ones we previously computed), spatial steps, and the coriolis factors.

The optimizer can be specified as a string (assuming it refers to an `optax` [common optimizers](https://optax.readthedocs.io/en/latest/api.html#)): `optim = "sgd"` for example.
Or designed using a more refined strategy:


```python
lr_scheduler = optax.exponential_decay(1e-2, 200, .5)
base_optim = optax.sgd(learning_rate=lr_scheduler)
base_optim = optax.sgd(learning_rate=lr_scheduler)
optim = optax.chain(optax.clip(1), base_optim)
```


```python
u_var, v_var, losses_var = cyclogeostrophy(u_geos, v_geos, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v, 
                                           return_losses=True)

norm_vorticity_var = tools.compute_norm_vorticity(u_var, v_var, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v)
norm_vorticity_var_t = tools.interpolate_south_west(norm_vorticity_var, axis=0, neuman=False)
norm_vorticity_var_t = tools.interpolate_south_west(norm_vorticity_var_t, axis=1, neuman=False)
```

#### Comparison to NEMO's velocities


```python
fig, axs = plt.subplots(2, 2, figsize=get_figsize(1, 20/6), 
                        subplot_kw={"projection": ccrs.PlateCarree()})

axs[0, 0].set_title("NEMO data")
_ = axs[0, 0].pcolormesh(lon, lat, norm_vorticity_t, cmap="RdBu_r", shading="auto", 
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

axs[0, 1].set_title("Variational cyclogeostrophy")
im1 = axs[0, 1].pcolormesh(lon, lat, norm_vorticity_var_t, cmap="RdBu_r", shading="auto", 
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())

_ = axs[1, 0].pcolormesh(lon, lat, magnitude, 
                         shading="auto", 
                         vmin=mmin, vmax=mmax,
                         transform=ccrs.PlateCarree())

im2 = axs[1, 1].pcolormesh(lon, lat, tools.compute_magnitude(u_var, v_var), 
                           shading="auto", 
                           vmin=mmin, vmax=mmax,
                           transform=ccrs.PlateCarree())

fig.tight_layout()
fig.subplots_adjust(right=0.64, wspace=0.01)

cbar_ax1 = fig.add_axes([0.65, 0.51, 0.01, 0.38])
_ = fig.colorbar(im1, cax=cbar_ax1)
cbar_ax1.set_title("$\\xi / f$")

cbar_ax2 = fig.add_axes([0.65, 0.05, 0.01, 0.38])
_ = fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$")
 
ax3 = fig.add_axes([0.73, 0.3, 0.27, 0.4])
ax3.set_title("Cyclogeostrophic disequilibrium - $J(\\vec{u}_c^{(n)})$")
ax3.set_xlabel("step")
ax3.set_ylabel("disequilibrium")
ax3.plot(losses_var[:np.argmin(losses_var)+10])

plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_20_0.png?raw=true)
   

### Iterative method

We use the same function, but with the argument `method="iterative"`. 


```python
u_iterative, v_iterative, losses_it = cyclogeostrophy(u_geos, v_geos, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v, 
                                                      method="iterative", return_losses=True)

norm_vorticity_iterative = tools.compute_norm_vorticity(u_iterative, v_iterative, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v)
norm_vorticity_iterative_t = tools.interpolate_south_west(norm_vorticity_iterative, axis=0, neuman=False)
norm_vorticity_iterative_t = tools.interpolate_south_west(norm_vorticity_iterative_t, axis=1, neuman=False)
```

#### Comparison to NEMO's velocities


```python
fig, axs = plt.subplots(2, 2, figsize=get_figsize(1, 20/6), 
                        subplot_kw={"projection": ccrs.PlateCarree()})

axs[0, 0].set_title("NEMO data")
_ = axs[0, 0].pcolormesh(lon, lat, norm_vorticity_t, cmap="RdBu_r", shading="auto", 
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

axs[0, 1].set_title("Iterative cyclogeostrophy")
im1 = axs[0, 1].pcolormesh(lon, lat, norm_vorticity_iterative_t, cmap="RdBu_r", shading="auto", 
                           vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())

_ = axs[1, 0].pcolormesh(lon, lat, magnitude, 
                         shading="auto", 
                         vmin=mmin, vmax=mmax,
                         transform=ccrs.PlateCarree())

im2 = axs[1, 1].pcolormesh(lon, lat, tools.compute_magnitude(u_iterative, v_iterative), 
                           shading="auto", 
                           vmin=mmin, vmax=mmax,
                           transform=ccrs.PlateCarree())

fig.tight_layout()
fig.subplots_adjust(right=0.64, wspace=0.01)

cbar_ax1 = fig.add_axes([0.65, 0.51, 0.01, 0.38])
_ = fig.colorbar(im1, cax=cbar_ax1)
cbar_ax1.set_title("$\\xi / f$")

cbar_ax2 = fig.add_axes([0.65, 0.05, 0.01, 0.38])
_ = fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$")
 
ax3 = fig.add_axes([0.73, 0.3, 0.27, 0.4])
ax3.set_title("Cyclogeostrophic disequilibrium - $J(\\vec{u}_c^{(n)})$")
ax3.set_xlabel("step")
ax3.set_ylabel("disequilibrium")
ax3.plot(losses_it)

plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_24_0.png?raw=true)
    

### Iterative method, with filter

We use the same function, but with the arguments `method="iterative"`, and `use_res_filter=True`. 


```python
u_it_filter, v_it_filter, losses_it_filter = cyclogeostrophy(u_geos, v_geos, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v, 
                                                             method="iterative", use_res_filter=True, return_losses=True)

norm_vorticity_it_filter = tools.compute_norm_vorticity(u_it_filter, v_it_filter, lat_u, lon_u, lat_v, lon_v, mask_u, mask_v)
norm_vorticity_it_filter_t = tools.interpolate_south_west(norm_vorticity_it_filter, axis=0, neuman=False)
norm_vorticity_it_filter_t = tools.interpolate_south_west(norm_vorticity_it_filter_t, axis=1, neuman=False)
```

#### Comparison to NEMO's currents


```python
fig, axs = plt.subplots(2, 2, figsize=get_figsize(1, 20/6), 
                        subplot_kw={"projection": ccrs.PlateCarree()})

axs[0, 0].set_title("NEMO data")
_ = axs[0, 0].pcolormesh(lon, lat, norm_vorticity_t, cmap="RdBu_r", shading="auto", 
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())

axs[0, 1].set_title("Iterative (filter) cyclogeostrophy")
im1 = axs[0, 1].pcolormesh(lon, lat, norm_vorticity_it_filter_t, cmap="RdBu_r", shading="auto", 
                           vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())

_ = axs[1, 0].pcolormesh(lon, lat, magnitude, 
                         shading="auto", 
                         vmin=mmin, vmax=mmax,
                         transform=ccrs.PlateCarree())

im2 = axs[1, 1].pcolormesh(lon, lat, tools.compute_magnitude(u_it_filter, v_it_filter), 
                           shading="auto", 
                           vmin=mmin, vmax=mmax,
                           transform=ccrs.PlateCarree())

fig.tight_layout()
fig.subplots_adjust(right=0.64, wspace=0.01)

cbar_ax1 = fig.add_axes([0.65, 0.51, 0.01, 0.38])
_ = fig.colorbar(im1, cax=cbar_ax1)
cbar_ax1.set_title("$\\xi / f$")

cbar_ax2 = fig.add_axes([0.65, 0.05, 0.01, 0.38])
_ = fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$")
 
ax3 = fig.add_axes([0.73, 0.3, 0.27, 0.4])
ax3.set_title("Cyclogeostrophic disequilibrium - $J(\\vec{u}_c^{(n)})$")
ax3.set_xlabel("step")
ax3.set_ylabel("disequilibrium")
ax3.plot(losses_it_filter)

plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_28_0.png?raw=true)
    

## Overall quantitative comparison


```python
percentiles = np.linspace(0, 1, 1000)
vorticity_percentile = np.quantile(norm_vorticity[~np.isnan(norm_vorticity)], percentiles)
vorticity_percentile_geos = np.quantile(norm_vorticity_geos[~np.isnan(norm_vorticity_geos)], percentiles)
vorticity_percentile_var = np.quantile(norm_vorticity_var[~np.isnan(norm_vorticity_var)], percentiles)
vorticity_percentile_iterative = np.quantile(norm_vorticity_iterative[~np.isnan(norm_vorticity_iterative)], percentiles)

fig = plt.figure(figsize=get_figsize(.5))
ax = fig.add_subplot(1, 1, 1)
ax.axline(xy1=(vorticity_percentile.min(), vorticity_percentile.min()), 
          xy2=(vorticity_percentile.max(), vorticity_percentile.max()), 
          linestyle="dashed", linewidth=1, color="black", label="NEMO data")
ax.scatter(vorticity_percentile, vorticity_percentile_geos, 
           s=1, label="Geostrophy")
ax.scatter(vorticity_percentile, vorticity_percentile_var, 
           s=1, label="Variational cyclogeostrophy")
ax.scatter(vorticity_percentile, vorticity_percentile_iterative, 
           s=1, label="Iterative cyclogeostrophy")
ax.legend()
ax.set_xlabel("NEMO data vorticity percentiles")
ax.set_ylabel("estimated vorticity percentiles")
ax.set_xscale('function', functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)), 
                                     lambda x: np.sign(x) * x**2))
ax.set_yscale('function', functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)), 
                                     lambda x: np.sign(x) * x**2))
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))

plt.show()
```
 
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/alboran_sea/output_30_0.png?raw=true)
