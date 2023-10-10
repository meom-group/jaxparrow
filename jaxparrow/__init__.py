import argparse

__version__ = "0.0.1"


def do_main(dir_data, name_mask, name_ssh, name_u, name_v, write_dir):
    return


def main():
    parser = argparse.ArgumentParser(prog="Cyclogeostrophic balance",
                                     description="Computes the inversion of the cyclogeostrophic balance using a "
                                                 "variational formulation with gradient descent minimization approach.")

    parser.add_argument("--dir_data", default="notebooks/data", type=str, help="data directory")
    parser.add_argument("--name_mask", default="mask_eNATL60MEDWEST_3.6.nc", type=str,
                        help="mask file name")
    parser.add_argument("--name_ssh", default="eNATL60MEDWEST-BLB002_y2009m07d01.1h_sossheig.nc",
                        type=str, help="SSH file name")
    parser.add_argument("--name_u", default="eNATL60MEDWEST-BLB002_y2009m07d01.1h_sozocrtx.nc",
                        type=str, help="u velocity file name")
    parser.add_argument("--name_v", default="eNATL60MEDWEST-BLB002_y2009m07d01.1h_somecrty.nc",
                        type=str, help="v velocity file name")
    parser.add_argument("--write_dir", default="notebooks/data", type=str,
                        help="cyclogeostrophic outputs directory")

    args = parser.parse_args()

    do_main(args.dir_data, args.name_mask, args.name_ssh, args.name_u, args.name_v, args.write_dir)
