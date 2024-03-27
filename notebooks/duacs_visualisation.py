from functools import partial

from cartopy import crs as ccrs
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def plot_ssh(
        ds,
        ti=0,  # time index
        longitude_name="longitude",  # variables names
        latitude_name="latitude",
        ssh_name="adt",
        label=None,
        ax=None
):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        
    im = ax.pcolormesh(
        ds[longitude_name], ds[latitude_name], ds[ssh_name][ti],
        cmap="turbo", shading="auto",
        vmin=ds[ssh_name].min(), vmax=ds[ssh_name].max(),
        transform=ccrs.PlateCarree()
    )
    
    clb = plt.colorbar(im, ax=ax)
    clb.ax.set_title("SSH (m)")
    
    ax.coastlines()
    if label is not None:
        ax.set_title(label)
        
    return im


def plot_currents(
        ds,
        ti=0,  # time index
        longitude_name="longitude",  # variables names
        latitude_name="latitude",
        magnitude_name="uv",
        vmin=None,
        vmax=None,
        label=None,
        ax=None
):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        
    if vmin is None:
        vmin=ds[magnitude_name].min()
    if vmax is None:
        vmax=ds[magnitude_name].max()
        
    im = ax.pcolormesh(
        ds[longitude_name], ds[latitude_name], ds[magnitude_name][ti],
        cmap="viridis", shading="auto",
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    clb = plt.colorbar(im, ax=ax)
    clb.ax.set_title("$\\vert\\vert \\vec{u} \\vert\\vert$ (m/s)")
    
    ax.coastlines()
    if label is not None:
        ax.set_title(label)
        
    return im


def update_ssh(im, ds, ti, ssh_name="adt"):
    im.set_array(ds[ssh_name][ti])
    return im


def update_currents(im, ds, ti, magnitude_name="uvgos"):
    im.set_array(ds[magnitude_name][ti])
    return im


class Animated:
    def __init__(
            self,
            ds,
            plot_vars,  # iterable, control what to plot
            init_fns,
            update_fns,
            ti0=None,  # starting time index
            ds_dt=np.timedelta64(1, "D"),  # temporal sampling (np.timedelta or similar)
            frame_dt=50,  # time interval between animation frames (in milliseconds)
            longitude_name="longitude",  # variables names
            latitude_name="latitude",
            time_name="time"
    ):
        if ti0 is None:
            ti0 = 0

        self.ds = ds
        self.plot_vars = plot_vars
        self.update_fns = update_fns

        time_da = ds[time_name]
        res_dt = time_da[1].data - time_da[0].data
        self.time = np.arange(time_da.min().data, time_da.max().data + res_dt, res_dt)
        t_step = 1
        if ds_dt > res_dt:  # resample if needed
            t_step = ds_dt // res_dt
        ti = np.arange(0, self.time.size, t_step)

        self.fig, ax = plt.subplots(1, len(self.plot_vars), figsize=(15, 4),
                                    subplot_kw={"projection": ccrs.PlateCarree()})

        self.artists = [
            init_fns[i](self.ds, ti0, longitude_name, latitude_name, self.plot_vars[i], ax=ax[i])
            for i in range(len(self.plot_vars))
        ]

        self.fig.suptitle(self.time[0].astype("datetime64[s]").item().strftime("%Y-%m-%d %H:%M"))
        self.fig.tight_layout()

        self.animation = FuncAnimation(
            fig=self.fig, func=self.update, blit=True,
            frames=ti, interval=frame_dt,
            cache_frame_data=False
        )

    def update(self, ti):
        artists = [
            self.update_fns[i](self.artists[i], self.ds, ti, self.plot_vars[i])
            for i in range(len(self.plot_vars))
        ]
        self.fig.suptitle(self.time[ti].astype("datetime64[s]").item().strftime("%Y-%m-%d %H:%M"))

        artists.append(self.fig)
        return artists


class AnimatedSSHCurrent(Animated):
    def __init__(
            self,
            ds,
            plot_vars,
            labels=(None, None),
            ti0=None,
            ds_dt=np.timedelta64(1, "D"),
            frame_dt=50,
            longitude_name="longitude",
            latitude_name="latitude",
            time_name="time"
    ):
        init_fns = [
            partial(plot_ssh, label=labels[0]),
            partial(plot_currents, label=labels[1]),
        ]
        update_fns = [update_ssh, update_currents]
        
        super().__init__(
            ds,
            plot_vars,
            init_fns,
            update_fns,
            ti0,
            ds_dt,
            frame_dt,
            longitude_name,
            latitude_name,
            time_name
        )


class AnimatedCurrents(Animated):
    def __init__(
            self,
            ds,
            plot_vars,
            labels,
            ti0=None,
            ds_dt=np.timedelta64(1, "D"),
            frame_dt=50,
            longitude_name="longitude",
            latitude_name="latitude",
            time_name="time"
    ):
        vmin = np.min([
            ds[plot_var].min() for plot_var in plot_vars
        ])
        vmax = np.max([
            ds[plot_var].max() for plot_var in plot_vars
        ])

        init_fns = [
            partial(plot_currents, vmin=vmin, vmax=vmax, label=labels[i]) for i in range(len(labels))
        ]
        update_fns = [update_currents] * len(labels)
        
        super().__init__(
            ds,
            plot_vars,
            init_fns,
            update_fns,
            ti0,
            ds_dt,
            frame_dt,
            longitude_name,
            latitude_name,
            time_name
        )
