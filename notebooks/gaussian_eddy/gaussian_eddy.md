```python
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxparrow.cyclogeostrophy import _iterative, _variational
from jaxparrow.geostrophy import _geostrophy
from jaxparrow.tools.kinematics import magnitude
from jaxparrow.tools.operators import interpolation
from jaxparrow.tools.sanitize import init_land_mask

sys.path.extend([os.path.join(os.path.dirname(os.getcwd()), "tests")])
from tests import gaussian_eddy as ge  # noqa

%reload_ext autoreload
%autoreload 2
```

# Method validation in the idealized gaussian eddy scenario

We want to use a gaussian eddy for our functional tests, as analytical solutions can be derived in that setting.

The gaussian eddy we consider is of the form $\eta = \eta_0 \exp^{-(r/R_0)^2}$, with $R_0$ its radius, $\eta_0$ the SSH anomaly at its center, and $r$ the radial distance. 
We choose to use a constant spatial step in meters.


```python
# Alboran sea settings
R0 = 50e3
ETA0 = .2
LAT = 36

dxy = 10e3
```

## Simulating the eddy


```python
X, Y, R, dXY, coriolis_factor, ssh, u_geos_t, v_geos_t, u_cyclo_t, v_cyclo_t, mask = ge.simulate_gaussian_eddy(
    R0, 
    dxy, 
    ETA0, 
    LAT
)
```

We just make sure that the grids are correct.


```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("X")
im = ax1.pcolormesh(X, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("Y")
im = ax2.pcolormesh(Y, shading="auto")
plt.colorbar(im, ax=ax2)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_6_0.png)
    



```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("R")
im = ax1.pcolormesh(X, Y, R, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("ssh")
im = ax2.pcolormesh(X, Y, ssh, shading="auto")
plt.colorbar(im, ax=ax2)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_7_0.png)
    


## Geostrophy

### Analytical

$$u_g = 2y \frac{g \eta_0}{f R_0^2} \exp^{-(r/R_0)^2} = 2y \frac{g \eta}{f R_0^2}$$

$$v_g = -2x \frac{g \eta_0}{f R_0^2} \exp^{-(r/R_0)^2} = -2x \frac{g \eta}{f R_0^2}$$


```python
azim_geos = magnitude(u_geos_t, v_geos_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_geos_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_geos_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_geos, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_geos_t[::5, ::5], v_geos_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_11_0.png)
    



```python
ax = plt.subplot()
ax.set_title("analytical geostrophy")
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azimuthal velocity (m/s)")
ax.scatter(R.flatten(), azim_geos.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_geos).flatten().argmax()], 
          ymin=azim_geos.min(), ymax=azim_geos.max(), colors="r", linestyles="dashed")
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_12_0.png)
    


### Numerical

$f\mathbf{k} \times \mathbf{u_g} = -g \nabla \eta$


```python
u_geos_est, v_geos_est = _geostrophy(ssh, dXY, dXY, coriolis_factor, mask)

u_geos_est_t = interpolation(u_geos_est, mask, axis=1, padding="left")
v_geos_est_t = interpolation(v_geos_est, mask, axis=0, padding="left")

azim_geos_est = magnitude(u_geos_est_t, v_geos_est_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_geos_est_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_geos_est_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_geos_est, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_geos_est_t[::5, ::5], v_geos_est_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_15_0.png)
    



```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("numerical geostrophy")
ax1.set_xlabel("radial distance (m)")
ax1.set_ylabel("azimuthal velocity (m/s)")
ax1.scatter(R.flatten(), azim_geos_est.flatten(), s=1)
ax1.vlines(R.flatten()[np.abs(azim_geos_est).flatten().argmax()], 
           ymin=azim_geos_est.min(), ymax=azim_geos_est.max(), colors="r", linestyles="dashed")
ax2.set_title("numerical error")
ax2.set_xlabel("radial distance (m)")
ax2.set_ylabel("absolute error (m/s)")
ax2.scatter(R.flatten(), azim_geos_est.flatten() - azim_geos.flatten(), s=1)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_16_0.png)
    



```python
ge.compute_rmse(u_geos_t, u_geos_est_t), ge.compute_rmse(v_geos_t, v_geos_est_t)
```




    (Array(0.0094964, dtype=float32), Array(0.00949639, dtype=float32))



## Cyclogeostrophic azimuthal velocity

### Analytical

$$V_{gr}=\frac{2V_g}{1+\sqrt{1+4V_g/(fr)}}$$

$$u_{gr} = u_g + sin(\theta) \frac{V_{gr}^2}{fr}$$
$$v_{gr} = v_g - cos(\theta) \frac{V_{gr}^2}{fr}$$


```python
azim_cyclo = magnitude(u_cyclo_t, v_cyclo_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_cyclo_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_cyclo_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_cyclo, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_cyclo_t[::5, ::5], v_cyclo_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_21_0.png)
    



```python
ax = plt.subplot()
ax.set_title("analytical cyclogeostrophy")
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azimuthal velocity (m/s)")
ax.scatter(R.flatten(), azim_cyclo.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_cyclo).flatten().argmax()], 
          ymin=azim_cyclo.min(), ymax=azim_cyclo.max(), colors="r", linestyles="dashed")
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_22_0.png)
    


### Numerical

$\mathbf{u} - \frac{\mathbf{k}}{f} \times (\mathbf{u} \cdot \nabla \mathbf{u}) = \mathbf{u_g}$


```python
u_geos_u = u_geos_est
v_geos_v = v_geos_est
mask = init_land_mask(u_geos_t)
```

#### Variational estimation


```python
optim = optax.sgd(learning_rate=5e-2)
u_cyclo_est, v_cyclo_est, _ = _variational(u_geos_u, v_geos_v, dXY, dXY, dXY, dXY,
                                           coriolis_factor, coriolis_factor, mask,
                                           n_it=20, optim=optim)

u_cyclo_est_t = interpolation(u_cyclo_est, mask, axis=1, padding="left")
v_cyclo_est_t = interpolation(v_cyclo_est, mask, axis=0, padding="left")

azim_cyclo_est = magnitude(u_cyclo_est_t, v_cyclo_est_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_cyclo_est, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_cyclo_est_t[::5, ::5], v_cyclo_est_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_27_0.png)
    



```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("variational cyclogeostrophy")
ax1.set_xlabel("radial distance (m)")
ax1.set_ylabel("azimuthal velocity (m/s)")
ax1.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax1.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()],
           ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
ax2.set_title("numerical error")
ax2.set_xlabel("radial distance (m)")
ax2.set_ylabel("absolute error (m/s)")
ax2.scatter(R.flatten(), azim_cyclo_est.flatten() - azim_cyclo.flatten(), s=1)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_28_0.png)
    



```python
ge.compute_rmse(u_cyclo_t, u_cyclo_est_t), ge.compute_rmse(v_cyclo_t, v_cyclo_est_t)
```




    (Array(0.00872519, dtype=float32), Array(0.00872519, dtype=float32))



### Iterative estimation

$\mathbf{u}^{(n+1)} = \mathbf{u_g} + \frac{\mathbf{k}}{f} \times (\mathbf{u}^{(n)} \cdot \nabla \mathbf{u}^{(n)})$

#### Ioannou

Use of a convolution filter when computing the residuals.


```python
u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v, dXY, dXY, dXY, dXY,
                                         coriolis_factor, coriolis_factor, mask,
                                         n_it=20, res_eps=0.01,
                                         use_res_filter=True, res_filter_size=3, 
                                         return_losses=False)

u_cyclo_est_t = interpolation(u_cyclo_est, mask, axis=1, padding="left")
v_cyclo_est_t = interpolation(v_cyclo_est, mask, axis=0, padding="left")

azim_cyclo_est = magnitude(u_cyclo_est_t, v_cyclo_est_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_cyclo_est, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_cyclo_est_t[::5, ::5], v_cyclo_est_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_33_0.png)
    



```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("iterative (filter) cyclogeostrophy")
ax1.set_xlabel("radial distance (m)")
ax1.set_ylabel("azimuthal velocity (m/s)")
ax1.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax1.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()], 
           ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
ax2.set_title("numerical error")
ax2.set_xlabel("radial distance (m)")
ax2.set_ylabel("absolute error (m/s)")
ax2.scatter(R.flatten(), azim_cyclo_est.flatten() - azim_cyclo.flatten(), s=1)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_34_0.png)
    



```python
ge.compute_rmse(u_cyclo_t, u_cyclo_est_t), ge.compute_rmse(v_cyclo_t, v_cyclo_est_t)
```




    (Array(0.0108008, dtype=float32), Array(0.0108008, dtype=float32))



#### Penven

No convolution filter, original approach.


```python
u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v, dXY, dXY, dXY, dXY,
                                         coriolis_factor, coriolis_factor, mask,
                                         n_it=20, res_eps=0.01, 
                                         use_res_filter=False, res_filter_size=1, 
                                         return_losses=False)

u_cyclo_est_t = interpolation(u_cyclo_est, mask, axis=1, padding="left")
v_cyclo_est_t = interpolation(v_cyclo_est, mask, axis=0, padding="left")

azim_cyclo_est = magnitude(u_cyclo_est_t, v_cyclo_est_t, interpolate=False)
```


```python
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax1.set_title("U")
im = ax1.pcolormesh(X, Y, u_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax1)
ax2.set_title("V")
im = ax2.pcolormesh(X, Y, v_cyclo_est_t, shading="auto")
plt.colorbar(im, ax=ax2)
ax3.set_title("azimuthal")
im = ax3.pcolormesh(X, Y, azim_cyclo_est, shading="auto")
plt.colorbar(im, ax=ax3)
ax3.quiver(X[::5, ::5], Y[::5, ::5], u_cyclo_est_t[::5, ::5], v_cyclo_est_t[::5, ::5], color='k')
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_38_0.png)
    



```python
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("iterative cyclogeostrophy")
ax1.set_xlabel("radial distance (m)")
ax1.set_ylabel("azimuthal velocity (m/s)")
ax1.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax1.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()], 
           ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
ax2.set_title("numerical error")
ax2.set_xlabel("radial distance (m)")
ax2.set_ylabel("absolute error (m/s)")
ax2.scatter(R.flatten(), azim_cyclo_est.flatten() - azim_cyclo.flatten(), s=1)
plt.show()
```


    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_39_0.png)
    



```python
ge.compute_rmse(u_cyclo_t, u_cyclo_est_t), ge.compute_rmse(v_cyclo_t, v_cyclo_est_t)
```




    (Array(0.01090674, dtype=float32), Array(0.01090674, dtype=float32))


