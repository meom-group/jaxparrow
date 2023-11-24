# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] jupyter={"outputs_hidden": false}
# The goal of this document is to open questions on some optimization results obtained when estimating cyclogeostrophic currents using a variational approa
#
# First, we briefly present the oceanographic background in Sections {ref}`sec:geostrophy` and {ref}`sec:cyclogeostrophy`. In Section {ref}`sec:cyclogeostrophy`, we also introduce our numerical resolution setting ({ref}`subsec:variational_approach`), and a potentially important aspect of our problem ({ref}`subsec:solution_existence`). Finally, Section {ref}`sec:optimization_questions` reveals some results, and our related questions. 

# %% [markdown] jupyter={"outputs_hidden": false}
# (sec:geostrophy)=
# # Geostrophic approximation
#
# Sea Surface Currents (SSC) can be easily approximated from satellite altimetry observations of the Sea Surface Height (SSH) using the geostrophic balance. 
# Geostrophy describes the balance between the pressure gradient force (indirectly observed via SSH), and the Coriolis force. Geostrophic currents satisfy this equilibrium:
#
# $$
# f \left(\vec{k} \times \vec{u}_g \right) = -g \nabla \eta,
# $$ (geostrophic_balance)
#
# where $f$ is the Coriolis parameter, $\vec{k}$ the vertical unit vector, $\vec{u}_g$ the geostrophic velocity, $g$ the gravity, and $\eta$ the SSH.

# %% [markdown] jupyter={"outputs_hidden": false}
# (sec:cyclogeostrophy)=
# # Cyclogeostrophic balance
#
# However, geostrophy alone is not always sufficient to accurately estimate SSC {cite}`penven2014cyclogeostrophic`, and the advective term $\vec{u} \cdot \nabla \vec{u}$ should be added back to the balance to take into consideration the centrifugial acceleration.
# Skipping some rearrangements, we can express the cyclogeostrophic balance as:
#
# $$
# \vec{u}_c - \frac{\vec{k}}{f} \times \left(\vec{u}_c \cdot \nabla \vec{u}_c \right) = \vec{u}_g, 
# $$ (cyclogeostrophic_balance)
#
# where $\vec{u}_c$ is the cyclogeostrophic velocity.
#
# (subsec:solution_existence)=
# ## Solution existence
#
# We know that, under some physically unstable conditions, Equation {eq}`cyclogeostrophic_balance` does not hold a mathematical solution {cite}`knox2006iterative`.
# It can be exhibited in the idealized scenario of a gaussian eddy.
#
# In that setting, the nonlinear term $\vec{u}_c \cdot \nabla \vec{u}_c$ simplifies to $-\frac{V_{gr}^2}{r} \vec{e}_r$, with $V_{gr}$ the azimuthal component of the velocity, $r$ the radial distance to the eddy center, and $\vec{e}_r$ the outward-directed radial unit vector. Consequently, Equation {eq}`cyclogeostrophic_balance` simplifies to the gradient-wind equation {cite}`knox2006iterative`:
#
# $$
# V_{gr} + \frac{V_{gr}^2}{fr} = V_g,
# $$ (gradient_wind)
#
# where $V_g$ is the azimuthal geostrophic velocity. From Equation {eq}`gradient_wind`, we obtain the "normal" physical solution:
#
# $$
# V_{gr} = \frac{2 V_g}{1 + \sqrt{1 + 4 V_g / (fr)}}
# $$ (gradient_wind_sol)
#
# By convention, in the northern hemisphere, $r$ is positive for cyclonic eddies, and negative for anticyclonic ones. Therefore, because of the square root in the denominator, there are no real solutions of $V_{gr}$ for small $r$ in anticyclonic eddies {cite}`knox2006iterative`. 
#
# (subsec:variational_approach)=
# ## Variational approach
#
# Because of the advective term $\vec{u}_c \cdot \nabla \vec{u}_c$, Equation {eq}`cyclogeostrophic_balance` is nonlinear, and solving it analytically is conceivable only in idealized scenarios (such as the gaussian one).
# We propose to solve it numerically by formulating the cyclogeostrophy as the variational problem:
#
# $$
# J(\vec{u}_c) = \left\lVert \vec{u_c} - \frac{\vec{k}}{f} \times \left(\vec{u_c} \cdot \nabla \vec{u_c}\right) - \vec{u_g} \right\rVert^2,
# $$ (var_functional)
#
# where $\lVert.\rVert$ is the discrete $L^2$ norm.
#
# ### jaxparrow
#
# Our Python package [`jaxparrow`](https://jaxparrow.readthedocs.io/en/latest/) implements this variational approach, leveraging JAX {cite}`bradbury2021jax`. Thanks to JAX automatic differentiation capabilities, $\nabla J$ is numerically available, and the cyclogeostrophic currents are estimated by minimizing Equation {eq}`var_functional` using a gradient-based optimizer, with $\vec{u}_c^{(0)} = \vec{u}_g$ as initial guess.

# %% [markdown] jupyter={"outputs_hidden": false}
# (sec:optimization_questions)=
# # Optimization related questions
#
# Our interrogations are based on some unexpected (to us) results we obtained when using `jaxparrow` with different optimizers for estimating cyclogeostrophic currents in the Alboran sea (a region of the Mediterranean sea).

# %% jupyter={"outputs_hidden": false}
# %%capture

# install jaxparrow and additional dependencies
# !pip install jaxparrow matplotlib

# %% jupyter={"outputs_hidden": false}
# %%capture

import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from myst_nb import glue
import numpy as np
import numpy.ma as ma
import xarray as xr

from jaxparrow.tools import compute_coriolis_factor, compute_spatial_step
from jaxparrow import cyclogeostrophy, geostrophy


# %% jupyter={"outputs_hidden": false}
# some utility functions

def abs_err(true: np.ma.masked_array, estimate: np.ma.masked_array, cutoff: float = .1) -> np.ma.masked_array:
    true = np.ma.copy(true)
    estimate = np.ma.copy(estimate)
    estimate[np.abs(true) < cutoff] = 0
    true[np.abs(true) < cutoff] = 0
    return estimate - true


def rel_err(true: np.ma.masked_array, estimate: np.ma.masked_array) -> np.ma.masked_array:
    return np.nan_to_num(abs_err(true, estimate, cutoff=0) / np.abs(true),
                         nan=0, neginf=0, posinf=0)


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


# %% [markdown] jupyter={"outputs_hidden": false}
# ## Alboran sea experiments
#
# The Alboran sea is an highly energetic (currents are strong) area where cyclogeostrophy is expected to better represent the ocean dynamic than geostrophy.
# We use plausible and realistic (but not real) data from the state-of-the-art NEMO ocean circulation model {cite}`nemo2022ocean` as reference.

# %% jupyter={"outputs_hidden": false}
# %%capture

# download, store, and extract NEMO netCDF files the directory data
# !wget -P data https: // ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/jaxparrow/alboransea.tar.gz
# !tar -xzf data/alboransea.tar.gz -C data
# !rm data/alboransea.tar.gz

# %% jupyter={"outputs_hidden": false}
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

# %% jupyter={"outputs_hidden": false}
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

# %% jupyter={"outputs_hidden": false}
# needed for computations
dx_ssh, dy_ssh = compute_spatial_step(lat_ssh, lon_ssh)
dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
dx_v, dy_v = compute_spatial_step(lat_v, lon_v)
coriolis_factor = compute_coriolis_factor(lat)
coriolis_factor_u = compute_coriolis_factor(lat_u)
coriolis_factor_v = compute_coriolis_factor(lat_v)

# %% [markdown] jupyter={"outputs_hidden": false}
# More precisely, we have access to simulated SSH and SSC velocities, and we derive SSC normalized vorticities ($\xi / f$) from the velocities (see {numref}`fig:nemo_data`).

# %% jupyter={"outputs_hidden": false}
norm_vorticity = compute_norm_vorticity(uvel, vvel, dy_u, dx_v, mask_ssh, coriolis_factor)

# %% jupyter={"outputs_hidden": false}
vmin = -4
vmax = -vmin

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 20/3))

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
im = ax3.pcolormesh(lon, lat, norm_vorticity, cmap="RdBu_r", shading="auto", 
                    vmin=vmin, vmax=vmax)
clb3 = plt.colorbar(im, ax=ax3)
clb3.ax.set_title("$\\xi / f$")

glue("fig:nemo_data", fig, display=False)
plt.close()  # comment for inline rendering

# %% [markdown] jupyter={"outputs_hidden": false}
# ```{glue:figure} fig:nemo_data
# :name: "fig:nemo_data"
#
# NEMO data
# ```

# %% [markdown] jupyter={"outputs_hidden": false}
# `jaxparrow` first estimates the geostrophic velocities (following Equation {eq}`geostrophic_balance`), from which we can computes the geostrophic vorticities (see {numref}`fig:geos`).

# %% jupyter={"outputs_hidden": false}
u_geos, v_geos = geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
u_geos = ma.masked_array(u_geos, mask_u)
v_geos = ma.masked_array(v_geos, mask_v)
norm_vorticity_geos = compute_norm_vorticity(u_geos, v_geos, dy_u, dx_v, mask_ssh, coriolis_factor)

# %% jupyter={"outputs_hidden": false}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figsize(2/3, 12.66/3))

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

glue("fig:geos", fig, display=False)
plt.close()  # comment for inline rendering

# %% [markdown] jupyter={"outputs_hidden": false}
# ```{glue:figure} fig:geos
# :name: "fig:geos"
#
# NEMO (left) and geostrophic (right) vorticities
# ```

# %% [markdown] jupyter={"outputs_hidden": false}
# Next, `jaxparrow` estimates the cyclogeostrophic velocities by minimizing Equation {eq}`var_functional`, with an optimizer chosen by the user. 

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Classical gradient descent
#
# Using gradient descent, we obtain very good qualitative results (see {numref}`fig:sgd`), with little sensitivity to the learning rate (not shown here).

# %% jupyter={"outputs_hidden": false}
# %%capture

u_var, v_var, losses_var = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, 
                                           optim="sgd", return_losses=True)
u_var = ma.masked_array(u_var, mask_u)
v_var = ma.masked_array(v_var, mask_v)
norm_vorticity_var = compute_norm_vorticity(u_var, v_var, dy_u, dx_v, mask_ssh, coriolis_factor)

# %% jupyter={"outputs_hidden": false}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 20/3))

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

glue("fig:sgd", fig, display=False)
plt.close()  # comment for inline rendering

# %% [markdown] jupyter={"outputs_hidden": false}
# ```{glue:figure} fig:sgd
# :name: "fig:sgd"
#
# NEMO (left) and SGD (right) vorticities
# ```

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Adam variation
#
# When using Adam, the loss curve suggests that we are still converging, but clearly, towards a qualitatively worse solution (see {numref}`fig:adam`).

# %% jupyter={"outputs_hidden": false}
# %%capture

u_var, v_var, losses_var = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, 
                                           optim="adam", return_losses=True)
u_var = ma.masked_array(u_var, mask_u)
v_var = ma.masked_array(v_var, mask_v)
norm_vorticity_var = compute_norm_vorticity(u_var, v_var, dy_u, dx_v, mask_ssh, coriolis_factor)

# %% jupyter={"outputs_hidden": false}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 20/3))

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

glue("fig:adam", fig, display=False)
plt.close()  # comment for inline rendering

# %% [markdown] jupyter={"outputs_hidden": false}
# ```{glue:figure} fig:adam
# :name: "fig:adam"
#
# NEMO (left) and Adam (right) vorticities
# ```

# %% [markdown] jupyter={"outputs_hidden": false}
# ## Open questions

# %% [markdown] jupyter={"outputs_hidden": false}
# ```{bibliography}
# :style: plain
# ```
