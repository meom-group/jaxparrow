import argparse
from datetime import datetime
import yaml

import numpy as np
import numpy.ma as ma
import xarray as xr

from .version import __version__
from .cyclogeostrophy import cyclogeostrophy


def _read_data(
        conf_path: str
) -> list:
    with open(conf_path) as f:
        conf = yaml.safe_load(f)  # parse conf file

    values = []
    seen_ds = {}

    # variable descriptions to be found in the conf file
    variables = ["ssh", "lon", "lat", "mask"]
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
    # and, optionally tune the output netCFD metadata attributes
    values.append(conf.get("out_attrs", {}))
    # finally, he must provide the full path (including its name and extension) of the output file
    values.append(conf["out_path"])

    return values


def _reverse_masks(
        mask_ssh: np.ndarray,
        mask_u: np.ndarray,
        mask_v: np.ndarray
) -> [np.ndarray, np.ndarray, np.ndarray]:
    def do_reverse(mask):
        if mask is not None:
            return 1 - mask
    return do_reverse(mask_ssh), do_reverse(mask_u), do_reverse(mask_v)


def _apply_masks(
        u_geos: np.ndarray,
        v_geos: np.ndarray,
        u_cyclo: np.ndarray,
        v_cyclo: np.ndarray,
        mask: np.ndarray
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def do_apply_mask(arr):
        if mask is not None:
            return ma.masked_array(arr, mask)
        else:
            return arr
    return (do_apply_mask(u_geos), do_apply_mask(v_geos),
            do_apply_mask(u_cyclo), do_apply_mask(v_cyclo))


def _create_attrs(
        conf_path: str,
        out_attrs: dict,
        run_datetime: str
) -> dict:
    with open(conf_path) as f:
        raw_conf = f.read()

    attrs = {
        "Conventions": "CF-1.10",
        "title": "ocean geostrophic and cyclogeostrophic currents",
        "institution": "",
        "source": "jaxparrow package - v" + __version__,
        "history": run_datetime + ": jaxparrow --conf_file " + conf_path,
        "references": "https://jaxparrow.readthedocs.io/",
        "comment": "jaxparrow computes the inversion of the cyclogeostrophic balance based on a variational formulation"
                   " approach, using JAX. ",
        "conf_content": raw_conf
    }
    attrs.update(out_attrs)

    return attrs


def _to_dataset(
        u_geos: ma.MaskedArray,
        v_geos: ma.MaskedArray,
        u_cyclo: ma.MaskedArray,
        v_cyclo: ma.MaskedArray,
        lat_u: ma.MaskedArray,
        lon_u: ma.MaskedArray,
        lat_v: ma.MaskedArray,
        lon_v: ma.MaskedArray,
        conf_path: str,
        out_attrs: dict,
        run_datetime: str
) -> xr.Dataset:
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
    }, attrs=_create_attrs(conf_path, out_attrs, run_datetime))
    return ds


def _write_data(
        u_geos: ma.MaskedArray,
        v_geos: ma.MaskedArray,
        u_cyclo: ma.MaskedArray,
        v_cyclo: ma.MaskedArray,
        lat_u: ma.MaskedArray,
        lon_u: ma.MaskedArray,
        lat_v: ma.MaskedArray,
        lon_v: ma.MaskedArray,
        conf_path: str,
        out_attrs: dict,
        run_datetime: str,
        out_path: str
):
    ds = _to_dataset(u_geos, v_geos, u_cyclo, v_cyclo, lat_u, lon_u, lat_v, lon_v, conf_path, out_attrs, run_datetime)
    ds.to_netcdf(out_path)


def _main(
        conf_path: str
):
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ssh_t, lon_t, lat_t, mask, cyclo_kwargs, out_attrs, out_path = _read_data(conf_path)

    mask = 1 - mask  # NEMO convention: 1 means marine area ; for ma.masked_array: 1 means masked (i.e. land)

    u_cyclo_u, v_cyclo_v, u_geos_u, v_geos_v, lat_u, lon_u, lat_v, lon_v = (
        cyclogeostrophy(ssh_t, lat_t, lon_t, mask, return_geos=True, **cyclo_kwargs))

    u_cyclo_u, v_cyclo_v, u_geos_u, v_geos_v = _apply_masks(u_cyclo_u, v_cyclo_v, u_geos_u, v_geos_v, mask)

    _write_data(u_cyclo_u, v_cyclo_v, u_geos_u, v_geos_v, lat_u, lon_u, lat_v, lon_v, conf_path, out_attrs,
                run_datetime, out_path)


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
