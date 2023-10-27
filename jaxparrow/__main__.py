import argparse
import os

import numpy as np
import numpy.ma as ma
import xarray as xr

from .tools import compute_coriolis_factor, compute_spatial_step
from .cyclogeostrophy import cyclogeostrophy
from .geostrophy import geostrophy


def _read_data(dir_data: str, name_mask: str, name_ssh: str, name_u: str, name_v: str) -> tuple:
    ds_mask = xr.open_dataset(os.path.join(dir_data, name_mask))
    mask_ssh = ds_mask.tmask[0, 0].values
    mask_u = ds_mask.umask[0, 0].values
    mask_v = ds_mask.vmask[0, 0].values

    ds_ssh = xr.open_dataset(os.path.join(dir_data, name_ssh))
    lon_ssh = ds_ssh.nav_lon.values
    lat_ssh = ds_ssh.nav_lat.values
    ssh = ds_ssh.sossheig[0].values

    ds_u = xr.open_dataset(os.path.join(dir_data, name_u))
    lon_u = ds_u.nav_lon.values
    lat_u = ds_u.nav_lat.values

    ds_v = xr.open_dataset(os.path.join(dir_data, name_v))
    lon_v = ds_v.nav_lon.values
    lat_v = ds_v.nav_lat.values

    return mask_ssh, mask_u, mask_v, ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v


def _apply_mask(mask_ssh: np.ndarray, mask_u: np.ndarray, mask_v: np.ndarray,
                ssh: np.ndarray, lon_ssh: np.ndarray, lat_ssh: np.ndarray,
                lon_u: np.ndarray, lat_u: np.ndarray, lon_v: np.ndarray, lat_v: np.ndarray) -> tuple:
    mask_ssh = 1 - mask_ssh
    mask_u = 1 - mask_u
    mask_v = 1 - mask_v

    ssh = ma.masked_array(ssh, mask_ssh)
    lon_ssh = ma.masked_array(lon_ssh, mask_ssh)
    lat_ssh = ma.masked_array(lat_ssh, mask_ssh)

    lon_u = ma.masked_array(lon_u, mask_u)
    lat_u = ma.masked_array(lat_u, mask_u)

    lon_v = ma.masked_array(lon_v, mask_v)
    lat_v = ma.masked_array(lat_v, mask_v)

    return ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v


def _compute_spatial_step(lon_ssh: np.ndarray, lat_ssh: np.ndarray, lon_u: np.ndarray, lat_u: np.ndarray,
                          lon_v: np.ndarray, lat_v: np.ndarray) -> tuple:
    dx_ssh, dy_ssh = compute_spatial_step(lat_ssh, lon_ssh)
    dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = compute_spatial_step(lat_v, lon_v)

    return dx_ssh, dy_ssh, dx_u, dy_u, dx_v, dy_v


def _compute_coriolis_factor(lat_u: np.ndarray, lat_v: np.ndarray) -> tuple:
    coriolis_factor_u = compute_coriolis_factor(lat_u)
    coriolis_factor_v = compute_coriolis_factor(lat_v)

    return coriolis_factor_u, coriolis_factor_v


def _to_dataset(u_geos: np.ndarray, v_geos: np.ndarray, u_cyclo: np.ndarray, v_cyclo: np.ndarray,
                lon_u: np.ndarray, lat_u: np.ndarray, lon_v: np.ndarray, lat_v: np.ndarray) -> xr.Dataset:
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
                lon_u: np.ndarray, lat_u: np.ndarray, lon_v: np.ndarray, lat_v: np.ndarray,
                out_dir: str, out_name: str):
    ds = _to_dataset(u_geos, v_geos, u_cyclo, v_cyclo, lon_u, lat_u, lon_v, lat_v)
    ds.to_netcdf(os.path.join(out_dir, out_name))


def _main(data_dir: str, name_mask: str, name_ssh: str, name_u: str, name_v: str, method: str,
          out_dir: str, out_name: str):
    mask_ssh, mask_u, mask_v, ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v = (
        _read_data(data_dir, name_mask, name_ssh, name_u, name_v))

    ssh, lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v = _apply_mask(mask_ssh, mask_u, mask_v, ssh, lon_ssh, lat_ssh,
                                                                    lon_u, lat_u, lon_v, lat_v)

    dx_ssh, dy_ssh, dx_u, dy_u, dx_v, dy_v = _compute_spatial_step(lon_ssh, lat_ssh, lon_u, lat_u, lon_v, lat_v)

    coriolis_factor_u, coriolis_factor_v = _compute_coriolis_factor(lat_u, lat_v)

    u_geos, v_geos = geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
    u_cyclo, v_cyclo = cyclogeostrophy(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                       method=method)

    if out_dir is None:
        out_dir = data_dir
    _write_data(u_geos, v_geos, u_cyclo, v_cyclo, lon_u, lat_u, lon_v, lat_v, out_dir, out_name)


def main():
    parser = argparse.ArgumentParser(prog="Cyclogeostrophic balance",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Computes the inversion of the cyclogeostrophic balance using a "
                                                 "variational or iterative approach.",
                                     epilog="""
 Inputs should be formatted as NEMO netCDF files:
    - mask file should contained \"tmask\", \"umask\" and \"vmask\" 4d (t, z, y, x) variables
    - ssh file should contained \"nav_lat\" and \"nav_lat\" 2d (y, x) variables (or coordinates), 
    and \"sossheig\" 3d (t, y, x) variable
    - u file should contained \"nav_lat\" and \"nav_lat\" 2d (y, x) variables (or coordinates)
    - v file should contained \"nav_lat\" and \"nav_lat\" 2d (y, x) variables (or coordinates)""")

    parser.add_argument("--data_dir", type=str, help="input data directory", required=True)
    parser.add_argument("--name_mask", type=str, help="mask file name", required=True)
    parser.add_argument("--name_ssh", type=str, help="SSH file name", required=True)
    parser.add_argument("--name_u", type=str, help="u lon/lat file name", required=True)
    parser.add_argument("--name_v", type=str, help="v lon/lat file name", required=True)
    parser.add_argument("--method", default="variational", type=str,
                        help="estimation method to use, one of \"variational\", \"penven\", \"ioannou\", defaults to "
                             "\"variational\"")
    parser.add_argument("--out_dir", type=str, help="cyclogeostrophic output directory")
    parser.add_argument("--out_name", default="out.nc", type=str,
                        help="cyclogeostrophic output file name")

    args = parser.parse_args()

    _main(args.data_dir, args.name_mask, args.name_ssh, args.name_u, args.name_v, args.method,
          args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
