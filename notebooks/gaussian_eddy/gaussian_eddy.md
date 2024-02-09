```python
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from jaxparrow.cyclogeostrophy import _iterative, _variational
from jaxparrow.geostrophy import _geostrophy

sys.path.extend([os.path.join(os.path.dirname(os.getcwd()), "tests")])
from tests import gaussian_eddy as ge
```

# Gaussian eddy

We want to use a gaussian eddy for our functional tests, as analytical solutions can be derived in that setting.

The gaussian eddy we consider is of the form $\eta = \eta_0 \exp^{-(r/R_0)^2}$, with $R_0$ its radius, $\eta_0$ the SSH anomaly at its center, and $r$ the radial distance. 
We choose to use a constant spatial step in meters.


```python
# Alboran sea settings
R0 = 50e3
ETA0 = .1
LAT = 36

dxy = 3e3
```

## Simulating the eddy


```python
X, Y, R, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo = ge.simulate_gaussian_eddy(R0, dxy, ETA0, LAT)
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

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_6_0.png?raw=true)
    

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

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_7_0.png?raw=true)
    

## Geostrophic azimuthal velocity

### Simulated

$$u_g = 2y \frac{g \eta_0}{f R_0^2} \exp^{-(r/R_0)^2} = 2y \frac{g \eta}{f R_0^2}$$

$$v_g = -2x \frac{g \eta_0}{f R_0^2} \exp^{-(r/R_0)^2} = -2x \frac{g \eta}{f R_0^2}$$


```python
azim_geos = ge.compute_azimuthal_magnitude(u_geos, v_geos)
u_geos_t, v_geos_t = ge.reinterpolate(u_geos, axis=1), ge.reinterpolate(v_geos, axis=0)
```

```python
ax = plt.subplot()
ax.set_title("geos")
im = ax.pcolormesh(X, Y, azim_geos, shading="auto")
plt.colorbar(im, ax=ax)
ax.quiver(X, Y, u_geos_t, v_geos_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_11_0.png?raw=true)
    


```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_geos (m/s)")
ax.scatter(R.flatten(), azim_geos.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_geos).flatten().argmax()], 
          ymin=azim_geos.min(), ymax=azim_geos.max(), colors="r", linestyles="dashed")
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_12_0.png?raw=true)
    

### Geostrophic balance

$f\mathbf{k} \times \mathbf{u_g} = -g \nabla \eta$


```python
u_geos_est, v_geos_est = _geostrophy(ssh, dXY, dXY, coriolis_factor, coriolis_factor)
azim_geos_est = ge.compute_azimuthal_magnitude(u_geos_est, v_geos_est)
u_geos_est_t, v_geos_est_t = ge.reinterpolate(u_geos_est, axis=1), ge.reinterpolate(v_geos_est, axis=0)
```


```python
vmax = np.max([azim_geos, azim_geos_est])
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.set_title("geos")
im = ax1.pcolormesh(X, Y, azim_geos, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax1)
ax1.quiver(X, Y, u_geos_t, v_geos_t, color='k')
ax2.set_title("geos_est")
im = ax2.pcolormesh(X, Y, azim_geos_est, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax2)
ax2.quiver(X, Y, u_geos_est_t, v_geos_est_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_15_0.png?raw=true)
    

```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_geos_est (m/s)")
ax.scatter(R.flatten(), azim_geos_est.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_geos_est).flatten().argmax()], 
          ymin=azim_geos_est.min(), ymax=azim_geos_est.max(), colors="r", linestyles="dashed")
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_16_0.png?raw=true)
    

```python
ge.compute_rmse(u_geos, u_geos_est), ge.compute_rmse(v_geos, v_geos_est)
```


    (0.0030672674, 0.0030672674)


## Cyclogeostrophic azimuthal velocity

### Gradient wind analytical solution

$$V_{gr}=\frac{2V_g}{1+\sqrt{1+4V_g/(fr)}}$$

$$u_{gr} = u_g + sin(\theta) \frac{V_{gr}^2}{fr}$$
$$v_{gr} = v_g - cos(\theta) \frac{V_{gr}^2}{fr}$$


```python
azim_cyclo = ge.compute_azimuthal_magnitude(u_cyclo, v_cyclo)
u_cyclo_t, v_cyclo_t = ge.reinterpolate(u_cyclo, axis=1), ge.reinterpolate(v_cyclo, axis=0)
```


```python
ax = plt.subplot()
ax.set_title("cyclo")
im = ax.pcolormesh(X, Y, azim_cyclo, shading="auto")
plt.colorbar(im, ax=ax)
ax.quiver(X, Y, u_cyclo_t, v_cyclo_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_21_0.png?raw=true)
    

```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_cyclo (m/s)")
ax.scatter(R.flatten(), azim_cyclo.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_cyclo).flatten().argmax()], 
          ymin=azim_cyclo.min(), ymax=azim_cyclo.max(), colors="r", linestyles="dashed")
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_22_0.png?raw=true)
    

### Variational estimation

$\mathbf{u} - \frac{\mathbf{k}}{f} \times (\mathbf{u} \cdot \nabla \mathbf{u}) = \mathbf{u_g}$


```python
u_cyclo_est, v_cyclo_est = _variational(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, 
                                        2000, "sgd", None, False)
azim_cyclo_est = ge.compute_azimuthal_magnitude(u_cyclo_est, v_cyclo_est)
u_cyclo_est_t, v_cyclo_est_t = ge.reinterpolate(u_cyclo_est, axis=1), ge.reinterpolate(v_cyclo_est, axis=0)
```


```python
vmax = np.max([azim_cyclo, azim_cyclo_est])
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.set_title("cyclo")
im = ax1.pcolormesh(X, Y, azim_cyclo, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax1)
ax1.quiver(X, Y, u_cyclo_t, v_cyclo_t, color='k')
ax2.set_title("cyclo_est")
im = ax2.pcolormesh(X, Y, azim_cyclo_est, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax2)
ax2.quiver(X, Y, u_cyclo_est_t, v_cyclo_est_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_25_0.png?raw=true)


```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_cyclo (m/s)")
ax.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()], 
          ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
plt.show()
```
    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_26_0.png?raw=true)
    

```python
ge.compute_rmse(u_cyclo, u_cyclo_est), ge.compute_rmse(v_cyclo, v_cyclo_est)
```


    (0.003272222, 0.003272222)


### Iterative estimation

$\mathbf{u}^{(n+1)} = \mathbf{u_g} + \frac{\mathbf{k}}{f} \times (\mathbf{u}^{(n)} \cdot \nabla \mathbf{u}^{(n)})$

#### Ioannou

Use of a convolution filter when computing the residuals.


```python
u_cyclo_est, v_cyclo_est = _iterative(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, None, None,
                                      20, 0.01, "same", True, 3, False)
azim_cyclo_est = ge.compute_azimuthal_magnitude(u_cyclo_est, v_cyclo_est)
u_cyclo_est_t, v_cyclo_est_t = ge.reinterpolate(u_cyclo_est, axis=1), ge.reinterpolate(v_cyclo_est, axis=0)
```


```python
vmax = np.max([azim_cyclo, azim_cyclo_est])
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.set_title("cyclo")
im = ax1.pcolormesh(X, Y, azim_cyclo, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax1)
ax1.quiver(X, Y, u_cyclo_t, v_cyclo_t, color='k')
ax2.set_title("cyclo_est")
im = ax2.pcolormesh(X, Y, azim_cyclo_est, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax2)
ax2.quiver(X, Y, u_cyclo_est_t, v_cyclo_est_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_31_0.png?raw=true)
    

```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_cyclo (m/s)")
ax.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()], 
          ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_32_0.png?raw=true)
    

```python
ge.compute_rmse(u_cyclo, u_cyclo_est), ge.compute_rmse(v_cyclo, v_cyclo_est)
```


    (0.003091015, 0.003091015)


#### Penven

No convolution filter, original approach.


```python
u_cyclo_est, v_cyclo_est = _iterative(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, None, None,
                                      20, 0.01, "same", False, 3, False)
azim_cyclo_est = ge.compute_azimuthal_magnitude(u_cyclo_est, v_cyclo_est)
u_cyclo_est_t, v_cyclo_est_t = ge.reinterpolate(u_cyclo_est, axis=1), ge.reinterpolate(v_cyclo_est, axis=0)
```


```python
vmax = np.max([azim_cyclo, azim_cyclo_est])
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.set_title("cyclo")
im = ax1.pcolormesh(X, Y, azim_cyclo, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax1)
ax1.quiver(X, Y, u_cyclo_t, v_cyclo_t, color='k')
ax2.set_title("cyclo_est")
im = ax2.pcolormesh(X, Y, azim_cyclo_est, shading="auto", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax2)
ax2.quiver(X, Y, u_cyclo_est_t, v_cyclo_est_t, color='k')
plt.show()
```

    
![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_36_0.png?raw=true)
   

```python
ax = plt.subplot()
ax.set_xlabel("radial distance (m)")
ax.set_ylabel("azim_cyclo (m/s)")
ax.scatter(R.flatten(), azim_cyclo_est.flatten(), s=1)
ax.vlines(R.flatten()[np.abs(azim_cyclo_est).flatten().argmax()], 
          ymin=azim_cyclo_est.min(), ymax=azim_cyclo_est.max(), colors="r", linestyles="dashed")
plt.show()
```

![png](https://github.com/meom-group/jaxparrow/blob/main/notebooks/gaussian_eddy/output_37_0.png?raw=true)
    

```python
ge.compute_rmse(u_cyclo, u_cyclo_est), ge.compute_rmse(v_cyclo, v_cyclo_est)
```

    (0.0030908077, 0.0030908077)
