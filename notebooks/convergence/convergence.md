---
jupytext:
  text_representation:
    extension: .md
    formats: ipynb,md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

The goal of this document is to open questions on some optimization results obtained when estimating cyclogeostrophic currents using a variational approach.

First, we briefly present the oceanographic background in Sections {ref}`sec:geostrophy` and {ref}`sec:cyclogeostrophy`. In Section {ref}`sec:cyclogeostrophy`, we also introduce our numerical resolution setting ({ref}`subsec:variational_approach`), and an aspect of our problem of potential interest ({ref}`subsec:solution_existence`). Finally, Section {ref}`sec:optimization_questions` reveals some results leading to questions of potential interest. 

+++

(sec:geostrophy)=
# Geostrophic approximation

Sea Surface Currents (SSC) can be easily approximated from satellite altimetry observations of the Sea Surface Height (SSH) using the geostrophic balance. 
Geostrophy describes the balance between the pressure gradient force (indirectly observed via SSH), and the Coriolis force. Geostrophic currents satisfy this equilibrium:

$$
f \left(\vec{k} \times \vec{u}_g \right) = -g \nabla \eta,
$$ (geostrophic_balance)

where $f$ is the Coriolis parameter, $\vec{k}$ the vertical unit vector, $\vec{u}_g$ the geostrophic velocity, $g$ the gravity, and $\eta$ the SSH.

+++

(sec:cyclogeostrophy)=
# Cyclogeostrophic balance

However, geostrophy alone is not always sufficient to accurately estimate SSC {cite}`penven2014cyclogeostrophic`, and the advective term $\vec{u} \cdot \nabla \vec{u}$ should be added back to the balance.
Skipping some rearrangements, we can express the cyclogeostrophic balance as:

$$
\vec{u}_c - \frac{\vec{k}}{f} \times \left(\vec{u}_c \cdot \nabla \vec{u}_c \right) = \vec{u}_g, 
$$ (cyclogeostrophic_balance)

where $\vec{u}_c$ is the cyclogeostrophic velocity.

(subsec:solution_existence)=
## Solution existence

We know that, under some physically unstable conditions, Equation {eq}`cyclogeostrophic_balance` does not hold a mathematical solution {cite}`knox2006iterative`.
It can be exhibited in the idealized scenario of a gaussian eddy.

In that setting, the nonlinear term $\vec{u}_c \cdot \nabla \vec{u}_c$ simplifies to $-\frac{V_{gr}^2}{r} \vec{e}_r$, with $V_{gr}$ the azimuthal component of the velocity, $r$ the radial distance to the eddy center, and $\vec{e}_r$ the outward-directed radial unit vector. Consequently, Equation {eq}`cyclogeostrophic_balance` simplifies to the gradient-wind equation {cite}`knox2006iterative`:

$$
V_{gr} + \frac{V_{gr}^2}{fr} = V_g,
$$ (gradient_wind)

where $V_g$ is the azimuthal geostrophic velocity. From Equation {eq}`gradient_wind`, we obtain the "normal" physical solution:

$$
V_{gr} = \frac{2 V_g}{1 + \sqrt{1 + 4 V_g / (fr)}}
$$ (gradient_wind_sol)

By convention, in the northern hemisphere, $r$ is positive for cyclonic eddies, and negative for anticyclonic ones. Therefore, because of the square root in the denominator, there are no real solutions of $V_{gr}$ for small $r$ in anticyclonic eddies {cite}`knox2006iterative`. 

(subsec:variational_approach)=
## Variational approach

Because of the advective term $\vec{u}_c \cdot \nabla \vec{u}_c$, Equation {eq}`cyclogeostrophic_balance` is nonlinear, and solving it analytically is conceivable only in idealized scenarios (such as the gaussian one).
We propose to solve it numerically by formulating the cyclogeostrophy as the variational problem:

$$
J(\vec{u}_c) = \left\lVert \vec{u_c} - \frac{\vec{k}}{f} \times \left(\vec{u_c} \cdot \nabla \vec{u_c}\right) - \vec{u_g} \right\rVert^2,
$$ (var_functional)

where $\lVert.\rVert$ is the discrete $L^2$ norm.

### jaxparrow

Our Python package [`jaxparrow`](https://jaxparrow.readthedocs.io/en/latest/) implements this variational approach, leveraging JAX {cite}`bradbury2021jax`. Thanks to JAX automatic differentiation capabilities, $\nabla J$ is numerically available, and the cyclogeostrophic currents are estimated by minimizing Equation {eq}`var_functional` using a gradient-based optimizer, with $\vec{u}_c^{(0)} = \vec{u}_g$ as initial guess.

+++

(sec:optimization_questions)=
# Optimization related questions

Our interrogations are based on some unexpected (to us) results we obtained when using `jaxparrow` with different optimizers for estimating cyclogeostrophic currents in the Alboran sea (a region of the Mediterranean sea).

```{code-cell} ipython3
:tags: [hide-input,hide-output]

!pip install jaxparrow matplotlib
```

```{code-cell} ipython3
:tags: [hide-input,hide-output]

from IPython.utils import io
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr

from jaxparrow.tools import compute_coriolis_factor, compute_spatial_step
from jaxparrow import cyclogeostrophy, geostrophy
```

```{code-cell} ipython3
:tags: [hide-input,hide-output]

# some utility functions

from jaxparrow.tools.tools import compute_derivative, interpolate
def compute_norm_vorticity(u: np.ma.masked_array, v: np.ma.masked_array, 
                           dy_u: np.ma.masked_array, dx_v: ma.masked_array, 
                           mask: np.ndarray, f: np.ma.masked_array) -> np.ma.masked_array:
    du_dy = compute_derivative(u, dy_u, axis=0)
    du_dy = interpolate(du_dy, axis=1)
    dv_dx = compute_derivative(v, dx_v, axis=1)
    dv_dx = interpolate(dv_dx, axis=0)
    return ma.masked_array(dv_dx - du_dy, mask) / f


def get_figsize(width_ratio, wh_ratio=1):
    dpi_ref = 100
    full_width_px = 1600
    
    fig_width = full_width_px / dpi_ref * width_ratio
    fig_height = fig_width / wh_ratio
    return fig_width, fig_height
```

## Alboran sea experiments

The Alboran sea is an highly energetic (currents are strong) area where cyclogeostrophy is expected to better represent the ocean dynamic than geostrophy.
We use plausible and realistic (but not real) data from the state-of-the-art NEMO ocean circulation model {cite}`nemo2022ocean` as reference.

```{code-cell} ipython3
:tags: [hide-input,hide-output]

# download, store, and extract NEMO netCDF files the directory data
!wget -P data https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/jaxparrow/alboransea.tar.gz
!tar -xzf data/alboransea.tar.gz -C data
!rm data/alboransea.tar.gz
```

```{code-cell} ipython3
:tags: [hide-input,hide-output]

# read data
data_dir = "data"
name_mask = "mask_alboransea.nc"
name_coord = "coordinates_alboransea.nc"
name_ssh = "alboransea_sossheig.nc"
name_u = "alboransea_sozocrtx.nc"
name_v = "alboransea_somecrty.nc"

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

```{code-cell} ipython3
:tags: [hide-input,hide-output]

# some data manipulations
mask_u = 1 - mask_u
mask_v = 1 - mask_v
mask_ssh = 1 - mask_ssh
uvel = ma.masked_array(uvel, mask_u)
vvel = ma.masked_array(vvel, mask_v)
ssh = ma.masked_array(ssh, mask_ssh)
lon_u = ma.masked_array(lon_u, mask_u)
lat_u = ma.masked_array(lat_u, mask_u)
lon_v = ma.masked_array(lon_v, mask_v)
lat_v = ma.masked_array(lat_v, mask_v)
lon_ssh = ma.masked_array(lon_ssh, mask_ssh)
lat_ssh = ma.masked_array(lat_ssh, mask_ssh)
```

```{code-cell} ipython3
:tags: [hide-input,hide-output]

# needed for computations
dx_ssh, dy_ssh = compute_spatial_step(lat_ssh, lon_ssh)
dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
dx_v, dy_v = compute_spatial_step(lat_v, lon_v)
coriolis_factor = compute_coriolis_factor(lat)
coriolis_factor_u = compute_coriolis_factor(lat_u)
coriolis_factor_v = compute_coriolis_factor(lat_v)
```

More precisely, we have access to simulated SSH and SSC velocities, and we derive SSC normalized vorticities ($\xi / f$) from the velocities (see {numref}`fig:nemo_data`).

Normalized vorticities smaller than -1 (depicted as intense blue in the bottom panels of {numref}`fig:nemo_data`) represent highly unstable conditions {cite}`shcherbina2013statistics`. In such cases, the mathematical solution for cyclogeostrophic balance is not expected to exist, as cyclogeostrophy is no longer a physically valid approximation.

```{code-cell} ipython3
:tags: [hide-input,hide-output]

norm_vorticity = compute_norm_vorticity(uvel, vvel, dy_u, dx_v, mask_ssh, coriolis_factor)
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    width: 66.7%
  figure:
    caption: |
      NEMO data
    name: fig:nemo_data
---

vmin = -4
vmax = -vmin

_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=get_figsize(2/3, 10.5/3/2))

ax1.set_title("Sea Surface Height")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, ssh, cmap="turbo", shading="auto")
clb1 = plt.colorbar(im, ax=ax1)

clb1.ax.set_title("SSH (m)")
ax2.set_title("Current velocity")
ax2.set_xlabel("longitude")
im = ax2.pcolormesh(lon, lat, np.sqrt(uvel**2 + vvel**2), shading="auto")
ax2.quiver(lon[::5, ::5], lat[::5, ::5], uvel[::5, ::5], vvel[::5, ::5], color="k")
clb2 = plt.colorbar(im, ax=ax2)
clb2.ax.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$ (m/s)")

ax3.set_title("Current normalized vorticity")
ax3.set_xlabel("longitude")
ax3.set_ylabel("latitude")
im = ax3.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb3 = plt.colorbar(im, ax=ax3)
clb3.ax.set_title("$\\xi / f$")

ax4.set_title("Unstable location")
ax4.set_xlabel("longitude")
im = ax4.pcolormesh(lon, lat, norm_vorticity<-1, cmap=colors.ListedColormap(["lightgrey", "darkblue"]), 
                    rasterized=True, vmin=0, vmax=1)
clb4 = plt.colorbar(im, ax=ax4, ticks=[0, 1])
clb4.ax.set_yticklabels(["False", "True"])
clb4.ax.set_title("$\\xi / f < -1$")

plt.tight_layout()
plt.show()
```

+++

`jaxparrow` first estimates the geostrophic velocities (following Equation {eq}`geostrophic_balance`), from which we can computes the geostrophic vorticities (see {numref}`fig:geos`).

```{code-cell} ipython3
:tags: [hide-input,hide-output]

u_geos, v_geos = geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
u_geos = ma.masked_array(u_geos, mask_u)
v_geos = ma.masked_array(v_geos, mask_v)
norm_vorticity_geos = compute_norm_vorticity(u_geos, v_geos, dy_u, dx_v, mask_ssh, coriolis_factor)
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    width: 66.7%
  figure:
    caption: |
      NEMO (left) and geostrophic (right) vorticities
    name: fig:geos
---

_, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figsize(2/3, 10.5/3))

ax1.set_title("NEMO data")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb1 = plt.colorbar(im, ax=ax1)
clb1.ax.set_title("$\\xi / f$")

ax2.set_title("Geostrophy")
ax2.set_xlabel("longitude")
im = ax2.pcolormesh(lon, lat, norm_vorticity_geos, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb2 = plt.colorbar(im, ax=ax2)
clb2.ax.set_title("$\\xi / f$")

plt.tight_layout()
plt.show()
```

+++

Next, `jaxparrow` estimates the cyclogeostrophic velocities by minimizing Equation {eq}`var_functional`, with an optimizer chosen by the user. 

+++

### Classical gradient descent

Using gradient descent, we obtain very good qualitative results (see {numref}`fig:sgd`), with little sensitivity to the learning rate (not shown here).

```{code-cell} ipython3
:tags: [hide-input]

with io.capture_output() as captured:
    u_var, v_var, losses_var = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, 
                                               optim="sgd", return_losses=True)
    u_var = ma.masked_array(u_var, mask_u)
    v_var = ma.masked_array(v_var, mask_v)
    norm_vorticity_var = compute_norm_vorticity(u_var, v_var, dy_u, dx_v, mask_ssh, coriolis_factor)

print("SGD final loss: {}".format(losses_var[-1]))
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: |
      NEMO (left) and SGD (right) vorticities
    name: fig:sgd
---

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 15/3))

ax1.set_title("NEMO data")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb1 = plt.colorbar(im, ax=ax1)
clb1.ax.set_title("$\\xi / f$")

ax2.set_title("Cyclogeostrophy - SGD")
ax2.set_xlabel("longitude")
im = ax2.pcolormesh(lon, lat, norm_vorticity_var, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb2 = plt.colorbar(im, ax=ax2)
clb2.ax.set_title("$\\xi / f$")

ax3.set_title("Cyclogeostrophic disequilibrium - $J(\\vec{u}_c^{(n)})$")
ax3.set_xlabel("step")
ax3.set_ylabel("disequilibrium")
ax3.plot(losses_var[:np.argmin(losses_var)+10])

plt.tight_layout()
plt.show()
```

+++

### Adam variation

When using Adam, the loss curve from {numref}`fig:adam` suggests that we are still converging, and the last evaluation of the cost function is smaller than for SGD. But, qualitatively, the solution appears to be much much worse than for SGD (see {numref}`fig:adam`).

```{code-cell} ipython3
:tags: [hide-input]

with io.capture_output() as captured:
    u_var, v_var, losses_var = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, 
                                               optim="adam", return_losses=True)
    u_var = ma.masked_array(u_var, mask_u)
    v_var = ma.masked_array(v_var, mask_v)
    norm_vorticity_var = compute_norm_vorticity(u_var, v_var, dy_u, dx_v, mask_ssh, coriolis_factor)

print("Adam final loss: {}".format(losses_var[-1]))
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: |
      NEMO (left) and Adam (right) vorticities
    name: fig:adam
---

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 15/3))

ax1.set_title("NEMO data")
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")
im = ax1.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb1 = plt.colorbar(im, ax=ax1)
clb1.ax.set_title("$\\xi / f$")

ax2.set_title("Cyclogeostrophy - Adam")
ax2.set_xlabel("longitude")
im = ax2.pcolormesh(lon, lat, norm_vorticity_var, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb2 = plt.colorbar(im, ax=ax2)
clb2.ax.set_title("$\\xi / f$")

ax3.set_title("Cyclogeostrophic disequilibrium - $J(\\vec{u}_c^{(n)})$")
ax3.set_xlabel("step")
ax3.set_ylabel("disequilibrium")
ax3.plot(losses_var[:np.argmin(losses_var)+10])

plt.tight_layout()
plt.show()
```

+++

## Open questions

These first observations lead us to believe that the optimization problem is probably not regular, and that several solutions exist.
We think that it could be related to physical unstable conditions resulting in the absence of mathematical solutions to Equation {eq}`cyclogeostrophic_balance` (as exhibited in Equation {eq}`gradient_wind_sol`).
This leads us to question if:
- some mathematical conditions on the solution could by found?
- the solution could be less dependent on the optimizer with some regularization of the loss function?

+++

```{bibliography}
:style: plain
```
