import argparse
from typing import Union
import yaml

import numpy as np
import numpy.ma as ma
import xarray as xr

from .tools import compute_coriolis_factor, compute_spatial_step
from .cyclogeostrophy import cyclogeostrophy
from .geostrophy import geostrophy


def _read_data(conf_path: str) -> list:
    with open(conf_path) as f:
        conf = yaml.safe_load(f)  # parse conf file

    values = []
    seen_ds = {}

    # variable descriptions to be found in the conf file
    variables = ["mask_ssh", "mask_u", "mask_v", "lon_ssh", "lat_ssh", "ssh", "lon_u", "lat_u", "lon_v", "lat_v"]
    for var in variables:
        try:
            conf_entry = conf[var]

            # each variable refers to a netCDF file (xarray Dataset)
            if conf_entry["file_path"] not in seen_ds:
                seen_ds[conf_entry["file_path"]] = xr.open_dataset(conf_entry["file_path"])
            ds = seen_ds[conf_entry["file_path"]]
            # and to a xarray Dataset variable
            val = ds[conf_entry["var_name"]]
            # optionally, the user can use indexing (if one needs to extract observation at a specific time for ex.)
            if "index" in conf_entry:
                idx = conf_entry["index"]
                val = val[{val.dims[i]: idx[i] for i in range(len(idx))}]
        except KeyError as e:
            if "mask" in var:  # mask variables are optional
                val = None
            else:
                raise e

        values.append(val)

    # in addition, the user can provide arguments passed to the cyclogeostrophic method
    values.append(conf.get("cyclogeostrophy", {}))
    # and he must provide the full path (including its name and extension) of the output file
    values.append(conf["out_path"])

    return values


def _apply_mask(mask_ssh: Union[np.ndarray, None], mask_u: Union[np.ndarray, None], mask_v: Union[np.ndarray, None],
                ssh: np.ndarray, lon_ssh: np.ndarray, lat_ssh: np.ndarray,
                lon_u: np.ndarray, lat_u: np.ndarray, lon_v: np.ndarray, lat_v: np.ndarray) -> tuple:
    def __do_apply(arr: np.ndarray, mask: Union[np.ndarray, None]) -> np.ndarray:
        if mask is None:
            mask = np.ones_like(arr)
        mask = 1 - mask  # don't forget to invert the masks (for ma.MaskedArray, True means invalid)
        return ma.masked_array(arr, mask)

    ssh = __do_apply(ssh, mask_ssh)
    lon_ssh = __do_apply(lon_ssh, mask_ssh)
    lat_ssh = __do_apply(lat_ssh, mask_ssh)

    lon_u = __do_apply(lon_u, mask_u)
    lat_u = __do_apply(lat_u, mask_u)

    lon_v = __do_apply(lon_v, mask_v)
    lat_v = __do_apply(lat_v, mask_v)

    return ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v


def _compute_spatial_step(lon_ssh: ma.MaskedArray, lat_ssh: ma.MaskedArray,
                          lon_u: ma.MaskedArray, lat_u: ma.MaskedArray,
                          lon_v: ma.MaskedArray, lat_v: ma.MaskedArray) -> tuple:
    dx_ssh, dy_ssh = compute_spatial_step(lat_ssh, lon_ssh)
    dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = compute_spatial_step(lat_v, lon_v)

    return dx_ssh, dy_ssh, dx_u, dy_u, dx_v, dy_v


def _compute_coriolis_factor(lat_u: ma.MaskedArray, lat_v: ma.MaskedArray) -> tuple:
    coriolis_factor_u = compute_coriolis_factor(lat_u)
    coriolis_factor_v = compute_coriolis_factor(lat_v)

    return coriolis_factor_u, coriolis_factor_v


def _to_dataset(u_geos: np.ndarray, v_geos: np.ndarray, u_cyclo: np.ndarray, v_cyclo: np.ndarray,
                lon_u: ma.MaskedArray, lat_u: ma.MaskedArray, lon_v: ma.MaskedArray, lat_v: ma.MaskedArray) \
        -> xr.Dataset:
    ds = xr.Dataset({
        "u_geos": (["y", "x"], u_geos),
        "v_geos": (["y", "x"], v_geos),
        "u_cyclo": (["y", "x"], u_cyclo),
        "v_cyclo": (["y", "x"], v_cyclo)
    }, coords={
        "u_lon": (["y", "x"], lon_u),
        "u_lat": (["y", "x"], lat_u),
        "v_lon": (["y", "x"], lon_v),
        "v_lat": (["y", "x"], lat_v),
    })
    return ds


def _write_data(u_geos: np.ndarray, v_geos: np.ndarray, u_cyclo: np.ndarray, v_cyclo: np.ndarray,
                lon_u: ma.MaskedArray, lat_u: ma.MaskedArray, lon_v: ma.MaskedArray, lat_v: ma.MaskedArray,
                out_path: str):
    ds = _to_dataset(u_geos, v_geos, u_cyclo, v_cyclo, lon_u, lat_u, lon_v, lat_v)
    ds.to_netcdf(out_path)


def _main(conf_path: str):
    mask_ssh, mask_u, mask_v, ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v, cyclo_kwargs, out_path = (
        _read_data(conf_path))

    ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v = _apply_mask(mask_ssh, mask_u, mask_v, ssh, lon_ssh, lat_ssh,
                                                                    lon_u, lat_u, lon_v, lat_v)

    dx_ssh, dy_ssh, dx_u, dy_u, dx_v, dy_v = _compute_spatial_step(lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v)

    coriolis_factor_u, coriolis_factor_v = _compute_coriolis_factor(lat_u, lat_v)

    u_geos, v_geos = geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
    u_cyclo, v_cyclo = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                       **cyclo_kwargs)

    _write_data(u_geos, v_geos, u_cyclo, v_cyclo, lon_u, lat_u, lon_v, lat_v, out_path)


def main():
    parser = argparse.ArgumentParser(prog="Cyclogeostrophic balance",
                                     description="Computes the inversion of the cyclogeostrophic balance using a "
                                                 "variational or iterative approach.",
                                     epilog="For an example yaml configuration file, see the documentation: "
                                            "https://meom-group.github.io/jaxparrow/description.html#as-an-executable")

    parser.add_argument("--conf_path", type=str, help="yaml configuration file path", required=True)

    args = parser.parse_args()

    _main(args.conf_path)


if __name__ == "__main__":
    main()
