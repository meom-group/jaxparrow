---
title: 'jaxparrow: a Python package solving the cyclogeostrophic balance using a variational formulation'
tags:
  - oceanography
  - cyclogeostrophic balance
  - variational formulation
  - satellite altimetry
  - circulation modeling
  - Python
  - JAX
authors:
  - name: Vadim BERTRAND
    equal-contrib: true
    affiliation: 1
  - name: Victor E V Z DE ALMEIDA
    equal-contrib: true
    affiliation: 1
  - name: Julien LE SOMMER
    equal-contrib: true
    affiliation: 1
  - name: Emmanuel COSME
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: Université Grenoble Alpes, France
    index: 1
date: 10 july 2023
bibliography: paper.bib
---

# Summary

Sea Surface Height (SSH) variations measured by satellite altimeters are widely used to estimate Sea Surface Currents (SSC) in oceanographic operational or research applications. The geostrophic balance approximation, which relates the pressure gradient, the current velocity, and the Coriolis force, is commonly employed to estimate SSC from SSH. It is known that under some configurations, the velocity advection term, neglected in the geostrophic formulation, should be included in the balance, leading to the cyclogeostrophic balance approximation. In general, solving the cyclogeostrophic balance can not be done analytically and numerical methods are needed. However, (1) existing iterative approaches are known to diverge, and ad-hoc methods are used to avoid local discontinuities; (2) publicly available, well maintained implementations are missing.

To overcome these limitations, we propose the Python package `jaxparrow`.`jaxparrow` formulates the cyclogeostrophic balance as a variational problem and solve it using a collection of well known optimizers. Its implementation heavily relies on JAX, the Python library bringing together automatic differentiation and just-in-time compilation, and the growing ecosystem around it. `jaxparrow` can be used as a package for an easy integration to existing oceanographic pipelines, or as a standalone executable working directly with NetCDF files.

# Statement of need

Sea Surface Currents (SSC) can be easily approximated from satellite altimetry observations of the Sea Surface Height (SSH) using the geostrophic balance. Geostrophy describes the balance between the pressure gradient force (indirectly observed via SSH), and the Coriolis force. Geostrophic currents satisfy this equilibrium:

\begin{equation}\label{eq:geostrophic_balance}
f \left(\vec{k} \times \vec{u}_g \right) = -g \nabla \eta,
\end{equation}

where $f$ is the Coriolis parameter, $\vec{k}$ the vertical unit vector, $\vec{u}_g$ the geostrophic velocity, $g$ the gravity, and $\eta$ the SSH.

The geostrophic equation represents a drastic approximation of the Navier-Stokes equation adapted to ocean dynamics. However, as discussed by  @bakun2006fronts, @charney1955gulf,and @maximenko2006mean, geostrophy alone is not always sufficient to accurately estimate SSC. In particular, @penven2014cyclogeostrophic showed that, in the highly energetic Mozambique Channel, the geostrophic approximation can produce errors of the order of 30% in velocity estimates, and that the advection term $\vec{u} \cdot \nabla \vec{u}$ was needed in the balance to reduce these errors. Considering a horizontal, stationary, and inviscid flow, the momentum equation linking SSC velocities $\vec{u}$ with SSH —through geostrophic velocities $\vec{u}_g$ from \autoref{eq:geostrophic_balance}— can be expressed as:

\begin{equation}\label{eq:cyclogeostrophic_balance}
\vec{u}_c - \frac{\vec{k}}{f} \times \left(\vec{u}_c \cdot \nabla \vec{u}_c \right) = \vec{u}_g, 
\end{equation}

where $\vec{u}_c$ is the cyclogeostrophic velocity.

Ocean data and services providers, such as Copernicus Marine Environment Monitoring Service [@taburet2019duacs], use geostrophic balance to estimate SSC from SSH. @cao2023global demonstrates that applying cyclogeostrophic corrections to the global ocean over a 25-years period results in significantly different estimates of SSC. Ocean products could therefore greatly benefit from a robust and open estimation method of cyclogeostrophic currents, which, to our knowledge, is not presently available.

# Numerical resolution of the cyclogeostrophic inverse problem

Because of the advective term $\vec{u}_c \cdot \nabla \vec{u}_c$, \autoref{eq:cyclogeostrophic_balance} is nonlinear, and solving it analytically is conceivable only in idealized scenarios, making numerical approaches essential. The current state-of-the-art method to solve the cyclogeostrophic equation is the iterative formulation introduced by @arnason1962higher and @endlich1961computation, which consists of reaching balance using the following iterative scheme:

\begin{equation}\label{eq:iterative_method}
\vec{u}_c^{(n+1)} = \vec{u}_g + \frac{\vec{k}}{f} \times \left( \vec{u}_c^{(n)} \cdot \nabla \vec{u}_c^{(n)} \right),
\end{equation}

with $\vec{u}_c^{(0)} = \vec{u}_g$. This approach is known to diverge since @arnason1962higher, and in practice [@penven2014cyclogeostrophic; @ioannou2019cyclostrophic] the residual $res = \vert \vec{u}_c^{(n+1)} - \vec{u}_c^{(n)} \vert$ is used to control point by point the iteration process. The iterative procedure is usually stopped when the residual locally falls below 0.01 m/s or starts to increase.

To avoid the local divergence issue of the iterative process, and its ad-hoc control, we propose to formulate the cyclogeostrophy as the variational problem:

\begin{equation}\label{eq:var_functional}
J(\vec{u}_c) = \left\lVert \vec{u_c} - \frac{\vec{k}}{f} \times \left(\vec{u_c} \cdot \nabla \vec{u_c}\right) - \vec{u_g} \right\rVert^2,
\end{equation}

where $\lVert.\rVert$ is the discrete $L^2$ norm. `jaxparrow` implements this approach, leveraging JAX [@bradbury2021jax]. Thanks to JAX automatic differentiation capabilities, $\nabla J$ is numerically available, and the cyclogeostrophic currents are estimated by minimizing \autoref{eq:var_functional} using a gradient-based optimizer, with $\vec{u}_c^{(0)} = \vec{u}_g$ as initial guess.

# Application to the Alboran sea

The Alboran sea is an energetic area of the Mediterranean sea. We demonstrate below the need to consider cyclogeostrophy in this region, and the benefit of the variational formulation implemented in `jaxparrow`. The data and results presented here can be found in the [Alboran sea notebook](https://github.com/meom-group/jaxparrow/blob/joss_paper/notebooks/alboran_sea.ipynb) hosted on GitHub.

We use SSH and SSC from the eNATL60 configuration [@brodeau2020enatm60; @uchida2022cloud] of the state-of-the-art NEMO ocean circulation model [@nemo2022ocean] as reference data. \autoref{fig:ref} shows SSH, SSC, and normalized relative vorticities in this region.

![Reference data: on the left and middle panels, SSH and SSC velocity (colored by the magnitude, with arrows giving the direction) simulated by NEMO; on the right, the corresponding normalized vorticity. \label{fig:ref}](fig/ref.png){width="100%"}

Using SSH, `jaxparrow` can first estimate the geostrophic SSC with \autoref{eq:geostrophic_balance}. As geostrophy is a major mechanism governing ocean dynamics, vorticities derived from those velocities present an overall similarity with the ones obtain from NEMO data. However, we can clearly identify irregular areas (around (-4, 35.5), (-3, 36), and (-2.5, 35.5) in (longitude, latitude) coordinates, see \autoref{fig:geostrophy}) where the geostrophic balance fails to accurately reconstruct SSC.

![The qualitative comparison between reference (left panel) and geostrophic (right panel) normalized vorticities reveals several regions with highly erroneous estimations. \label{fig:geostrophy}](fig/geostrophy.png){width="66.7%"}

Starting from geostrophic currents, `jaxparrow` solves the variational formulation of the cyclogeostrophy (\autoref{eq:var_functional}), using in this example the classical gradient descent [@kantorovich2016functional]. As a result, almost all the problematic areas are now much more accurately reconstructed, leaving mainly costal or domain boundary regions with large differences from our reference vorticity (see \autoref{fig:variational}, left and middle panels). By evaluating the cyclogeostrophic disequilibrium (the functional $J$ in \autoref{eq:var_functional}) along the optimization process, we observe that `jaxparrow` iteratively converges towards cyclogeostrophic balance (right panel of \autoref{fig:variational}).

![In contrast to the geostrophic approximation, variational cyclogeostrophy (middle panel) provide accurate reconstruction of the reference (left panel) normalized vorticities. The right panel demonstrates the fast convergence towards cyclogeostrophic balance. \label{fig:variational}](fig/variational.png){width="100%"}

For comparison, `jaxparrow` can also estimate the cyclogeostrophic currents using the iterative scheme (\autoref{eq:iterative_method}). In this example, the evolution of $J$ at each iteration reveals that this approach is not able to fill the cyclogeostrophic balance (see \autoref{fig:iterative}, right panel). We even notice that this estimation is qualitatively worse than the geostrophy (left and middle panels of \autoref{fig:iterative}).

![As exhibited in the right panel, the iterative approach diverges from the cyclogeostrophic balance; and we can notice from the two other panels that the resulting normalized vorticity is qualitatively worse than the geostrophic one. \label{fig:iterative}](fig/iterative.png){width="100%"}

Those qualitative observations are supported by more quantitative analysis. We computed the 1000 first percentiles of the vorticity distributions, and we observe, via a Q-Q plot [@wilk1968probability], that the percentiles of the variational distribution are the closest to the ones of the reference distribution (\autoref{fig:qqplot}).

![The percentiles of the normalized vorticity distributions demonstrate that our variational estimation of the cyclogeostrophy (in orange) corrects the geostrophy approximation (in blue), while the iterative scheme (in green) tends to diverge from the reference (in black). \label{fig:qqplot}](fig/qqplot.png){width="50%"}

# Availability

Beside the novel variational formulation, `jaxparrow` also offers the first to our knowledge open implementation of the cyclogeostrophy inversion. The code is available on [GitHub](https://github.com/meom-group/jaxparrow), with the specific tag `joss` for the version matching this publication; and the documentation, with pip-installation instructions, usage examples, and toy notebooks, is hosted on [Read the Docs](https://jaxparrow.readthedocs.io).

# References
