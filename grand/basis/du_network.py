"""
Handling Detector Unit (DU) network, footprint plots
"""
from logging import getLogger

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider
from matplotlib.offsetbox import AnchoredText
from matplotlib.backend_bases import MouseButton
from scipy.spatial import Delaunay


logger = getLogger(__name__)


def closest_node(node, nodes):  # pragma: no cover
    """Simple computation of distance mouse DU

    :param node:
    :param nodes:
    """
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


class DetectorUnitNetwork:
    """
    Handling a DU network

    Public attributs:

        * name str: name of the set of trace
        * du_pos float(nb_du, 3): position of DU
        * idx2idt int(nb_du): array of identifier of DU
    """

    def __init__(self, name="NotDefined"):
        self.name = name
        nb_du = 0
        self.du_pos = np.zeros((nb_du, 3))
        self.idx2idt = np.arange(nb_du)
        self.area_km2 = -1

    def init_pos_id(self, du_pos, du_id=None):
        """Init object with array DU position and identifier

        :param du_pos: position of DU
        :type du_pos: float[nb_DU, 3]
        :param du_id: identifier of DU
        :type du_id: list or array of string
        """
        if du_id is None:
            du_id = list(range(du_pos.shape[0]))
        self.du_pos = du_pos
        self.area_km2 = -1
        self.idx2idt = du_id
        assert isinstance(self.du_pos, np.ndarray)
        assert isinstance(self.idx2idt, (list, np.ndarray))
        assert du_pos.shape[0] == len(du_id)
        assert du_pos.shape[1] == 3

    def keep_only_du_with_index(self, l_idx):
        """Keep DU at index defined in list <l_idx>

        :param l_idx: list of index of DU
        """
        du_id = [self.idx2idt[idx] for idx in l_idx]
        self.idx2idt = du_id
        self.du_pos = self.du_pos[l_idx]
        self.area_km2 = -1

    def reduce_nb_du(self, new_nb_du):
        """Feature to debug and reduce computation

        :param new_nb_du: keep only new_nb_du first DU
        :type new_nb_du: int
        """
        self.idx2idt = self.idx2idt[:new_nb_du]
        self.du_pos = self.du_pos[:new_nb_du, :]
        self.area_km2 = -1

    def get_sub_network(self, l_id):
        """Reduce networh to DU in list <l_id>

        :param l_id: list of DU slected
        :type: int[nb_DU in l_id]
        """
        sub_net = DetectorUnitNetwork("sub-network of " + self.name)
        sub_net.init_pos_id(self.du_pos[l_id], self.idx2idt[l_id])
        return sub_net

    def get_nb_du(self):
        """Return the number of DU"""
        return len(self.idx2idt)

    def get_surface(self):
        """Return suface in km2

        :return: [km2] surface of network envelop
        :rtype: float
        """
        if self.area_km2 >= 0:
            return self.area_km2
        if self.du_pos.shape[0] < 3:
            self.area_km2 = 0
        pts = self.du_pos[:, :2].astype(np.float64)
        self.delaunay = Delaunay(self.du_pos[:, :2])
        triangle = self.delaunay.simplices
        a_area = np.abs(
            np.cross(
                pts[triangle[:, 1], :] - pts[triangle[:, 0], :],
                pts[triangle[:, 2], :] - pts[triangle[:, 0], :],
            )
        )
        a_area /= 2
        self.area_km2 = np.sum(a_area) / 1e6
        return self.area_km2

    def get_max_dist_du(self):
        """TODO
        :return: [km] distance max between two DU of network
        :rtype: float
        """
        # TODO:
        raise NotImplementedError

    ### PLOT

    def plot_du_pos(self):  # pragma: no cover
        """Plot DU position"""
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_footprint_1d(
        self, a_values, title="", traces=None, scale="log", unit=""
    ):  # pragma: no cover
        """Interactive footprint double click on DU draw trace associated and power spectrum


        :param a_values: intensity associated to DU
        :type a_values: float (nb DU)
        :param title: title of figure
        :type title: str
        :param traces: object traces
        :type traces: Handling3dTraces
        :param scale: type of scale
        :type scale: str in ["log", "lin"]

        """
        size_circle = 200
        cur_idx_plot = -1

        def on_move(event):
            nonlocal cur_idx_plot
            if event.inaxes:
                idx = closest_node(np.array([event.xdata, event.ydata]), self.du_pos[:, :2])
                if idx != cur_idx_plot:
                    cur_idx_plot = idx
                    anch_du.txt.set_text(f"DU={self.idx2idt[idx]}")
                    if scale == "lin":
                        anch_val.txt.set_text(f"{a_values[idx]:.2f}")
                    else:
                        anch_val.txt.set_text(f"{a_values[idx]:.2e}")
                    plt.draw()

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                # logger.info(f"on_click {event.xdata}, {event.ydata}")
                if traces:
                    traces.plot_trace_idx(cur_idx_plot)
                    traces.plot_psd_trace_idx(cur_idx_plot)
                    plt.show()
                plt.draw()

        fig, ax1 = plt.subplots(1, 1)
        s_title = f"{title}\n{self.get_nb_du()} DUs; Surface {int(self.get_surface())} km$^2$"
        s_title += f"; Name site: {self.name}"
        ax1.set_title(s_title)
        vmin = np.nanmin(a_values)
        vmax = np.nanmax(a_values)
        norm_user = colors.LogNorm(vmin=vmin, vmax=vmax)
        if scale == "log":
            my_cmaps = "Reds"
        elif scale == "lin":
            norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
            my_cmaps = "Blues"
        else:
            logger.error(f'scale must be in ["log","lin"]')
        scm = ax1.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=norm_user,
            s=size_circle,
            c=a_values,
            edgecolors="k",
            cmap=my_cmaps,
        )
        fig.colorbar(scm, label=unit)
        xlabel = "North [m]  (azimuth=0°) =>"
        if traces is not None:
            xlabel += f"\n{traces.info_shower}"
            xlabel += f"\n{traces.name}"
        plt.xlabel(xlabel)
        plt.ylabel(rf"West [m]  (azimuth=+90°) =>")
        ax1.grid()
        anch_du = AnchoredText("DU id", prop=dict(size=10), frameon=False, loc="upper left")
        anch_val = AnchoredText("Value", prop=dict(size=10), frameon=False, loc="upper right")
        ax1.axis("equal")
        ax1.add_artist(anch_du)
        ax1.add_artist(anch_val)
        plt.connect("motion_notify_event", on_move)
        if traces:
            plt.connect("button_press_event", on_click)

    def plot_footprint_4d(
        self, o_tr, v_plot, title="", same_scale=True, unit=""
    ):  # pragma: no cover
        """Plot footprint of time max by DU and value max by component

        :param o_tr: object traces
        :type o_tr: Handling3dTraces
        :param title: title of plot
        :type title: str
        """

        def subplot(plt_axis, a_values, cpnt="", scale="log"):
            ax1 = plt_axis
            size_circle = 80

            ax1.set_title(cpnt)
            if isinstance(scale, str):
                my_cmaps = "Blues"
                vmin = np.nanmin(a_values)
                vmax = np.nanmax(a_values)
                norm_user = colors.LogNorm(vmin=vmin, vmax=vmax)
                if scale == "log":
                    pass
                elif scale == "lin":
                    norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    logger.error(f'scale must be in ["log","lin"]')
            else:
                norm_user = scale
                my_cmaps = "Reds"
            scm = ax1.scatter(
                self.du_pos[:, 0],
                self.du_pos[:, 1],
                norm=norm_user,
                s=size_circle,
                c=a_values,
                edgecolors="k",
                cmap=my_cmaps,
            )
            ax1.axis("equal")
            ax1.grid()
            # plt.ylabel("[m]")
            # plt.xlabel("[m]")
            return scm

        fig, ax2 = plt.subplots(2, 2)

        t_max, _ = o_tr.get_tmax_vmax()
        ret_scat = subplot(ax2[0, 0], t_max, cpnt="Time of max value", scale="lin")
        fig.colorbar(ret_scat, label="ns")
        # same scale for
        if same_scale:
            vmin = np.nanmin(v_plot)
            vmax = np.nanmax(v_plot)
            norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm_user = "lin"
        ret_scat = subplot(ax2[1, 0], v_plot[:, 0], f"{title} {o_tr.axis_name[0]}", norm_user)
        fig.colorbar(ret_scat, label=unit)
        ret_scat = subplot(ax2[0, 1], v_plot[:, 1], f"{title} {o_tr.axis_name[1]}", norm_user)
        fig.colorbar(ret_scat, label=unit)
        ret_scat = subplot(ax2[1, 1], v_plot[:, 2], f"{title} {o_tr.axis_name[2]}", norm_user)
        fig.colorbar(ret_scat, label=unit)

    def plot_footprint_time(self, a_time, a3_values, title=""):  # pragma: no cover
        """Interactive plot, footprint max value for time defined by user with slider widget

        :param a_time: array of complet time of event
        :type a_time: float[nb_sample full event]
        :param a3_values: trace 3D
        :type a3_values: float [nb_DU, 3]
        :param title: title of plot
        :type title: str
        """
        # same number of sample
        assert a_time.shape[0] == a3_values.shape[2]
        # we plot norm of 3D vector
        a_norm_val = np.linalg.norm(a3_values, axis=1)
        val_min = a_norm_val.min()
        if val_min <= 0:
            val_min = 0.01
        col_log = colors.LogNorm(vmin=val_min, vmax=a_norm_val.max(), clip=True)
        # col_log = colors.LogNorm(clip=True)
        cmap_b = matplotlib.cm.get_cmap("Blues")
        delta_t = a_time[1] - a_time[0]
        # Create the figure and the line that we will manipulate
        fig, ax1 = plt.subplots()
        ax1.set_title(title)
        scat = ax1.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=col_log,
            s=200,
            c=a_norm_val[:, 0],
            edgecolors="k",
            cmap=cmap_b,
        )
        ax1.axis("equal")
        ax1.grid()
        fig.colorbar(scat)
        plt.ylabel("meters,          West (azimuth=90°) =>")
        plt.xlabel("meters,          North =>")
        fig.subplots_adjust(left=0.2, bottom=0.2)
        # Make a horizontal slider to control the frequency.
        axe_idx = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        time_slider = Slider(
            ax=axe_idx,
            label="time ns:",
            valmin=float(a_time[0]),
            valmax=float(a_time[-1]),  # a_time.shape[0]-1,
            valinit=float(a_time[0]),
        )

        # The function to be called anytime a slider's value changes
        def update_time(t_slider):
            frame_number = int((t_slider - a_time[0]) / delta_t)
            scat.set_array(a_norm_val[:, frame_number])
            return scat

        time_slider.on_changed(update_time)
        # plt.connect("button_press_event", on_click)
        # WARNING: we must used plt.show() of this module not in another module
        #          else slider is blocked
        plt.show()
