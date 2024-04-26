"""
REFERENCES
----------
.. [1] NEMO convention https://www.nemo-ocean.eu/doc/node19.html
.. [2] Penven et al. (2014) https://doi.org/10.1002/2013JC009528
.. [3] Ioannou et al. (2019) https://doi.org/10.1029/2019JC015031
.. [4] Optax optimizers https://optax.readthedocs.io/en/latest/api/optimizers.html
.. [5] Arbic et al. (2012) https://doi.org/10.1029/2011JC007367
.. [6] Fornberg (2021) https://doi.org/10.1093/imanum/draa006
"""

from .__main__ import main  # noqa
from .cyclogeostrophy import cyclogeostrophy
from .geostrophy import geostrophy
from .version import __version__  # noqa

__all__ = ["cyclogeostrophy", "geostrophy"]
