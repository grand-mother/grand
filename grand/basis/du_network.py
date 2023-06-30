"""
Handling DU network, footprint plot
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


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


class DetectorUnitNetwork:
    """
    Handling a DU network

    Public attributs:

        * name str: name of the set of trace
        * du_pos float(nb_du, 3): position of DU
        * du_id int(nb_du): array of identifier of DU
        * t_start_ns float(nb_du): time of first sample of trace

    """

    def __init__(self, name="NotDefined"):
        self.name = name
        nb_du = 0
        self.du_pos = np.zeros((nb_du, 3))
        self.du_id = np.arange(nb_du)

    def init_pos_id(self, du_pos, du_id):
        """
        Init object with array position and identifier

        :param du_pos: position of DU
        :type du_pos: float[nb_DU, 3]
        :param du_id: identifier of DU
        :type du_id: int[nb_DU]
        """
        self.du_pos = du_pos
        self.du_id = du_id
        assert isinstance(self.du_pos, np.ndarray)
        assert isinstance(self.du_id, np.ndarray)
        assert du_pos.shape[0] == du_id.shape[0]
        assert du_pos.shape[1] == 3

    def reduce_nb_du(self, new_nb_du):
        """
        Feature to debug and reduce computation

        :param new_nb_du: keep only new_nb_du first DU
        :type new_nb_du: int
        """
        self.du_id = self.du_id[:new_nb_du]
        self.du_pos = self.du_pos[:new_nb_du, :]

    def get_sub_network(self, l_id):
        """
        Reduce networh to DU in list l_id

        :param l_id: list of DU slected
        :type: int[nb_DU in l_id]
        """
        sub_net = DetectorUnitNetwork("sub-network of " + self.name)
        sub_net.init_pos_id(self.du_pos[l_id], self.du_id[l_id])
        return sub_net

    def get_pos_id(self, l_id):
        """
        :param l_id:
        :type l_id:
        """
        # TODO:
        raise NotImplementedError

    def get_surface(self):
        """
        Return suface in km2

        :return: [km2] surface of network envelop
        :rtype: float
        """
        # TODO:
        pts = self.du_pos[:, :2].astype(np.float64)
        self.delaunay = Delaunay(self.du_pos[:, :2])
        triangle = self.delaunay.simplices
        print(triangle[:20])
        a_area = np.abs(
            np.cross(
                pts[triangle[:, 1], :] - pts[triangle[:, 0], :],
                pts[triangle[:, 2], :] - pts[triangle[:, 0], :],
            )
        )
        a_area /= 2
        self.area_km2 = np.sum(a_area) / 1e6
        # self.sigma_norm_area = a_area.std() / a_area.mean()
        # plt.hist(a_area)
        # print(a_area[:20])
        # print(a_area.std())
        return self.area_km2

    def get_max_dist_du(self):
        """
        :return: [km] distance max between two DU of network
        :rtype: float
        """
        # TODO:
        raise NotImplementedError

    ### PLOT

    def plot_du_pos(self):  # pragma: no cover
        """
        plot DU position
        """
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_footprint_1d(self, a_values, title="", traces=None, scale="log"):  # pragma: no cover
        """
        Interactive footprint double click on DU draw trace associated and power spectrum


        :param a_values: intensity associated to DU
        :type a_values: float (nb DU)
        :param title: title of figure
        :type title: str
        :param traces: object traces
        :type traces: Handling3dTracesOfEvent
        :param scale: type of scale
        :type scale: str in ["log", "lin"]

        """
        size_circle = 200
        cur_idx_plot = -1

        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            return np.argmin(dist_2)

        def on_move(event):
            nonlocal cur_idx_plot
            if event.inaxes:
                idx = closest_node(np.array([event.xdata, event.ydata]), self.du_pos[:, :2])
                if idx != cur_idx_plot:
                    cur_idx_plot = idx
                    anch_du.txt.set_text(f"DU={self.du_id[idx]}")
                    anch_val.txt.set_text(f"{a_values[idx]:.2e}")
                    plt.draw()

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                # logger.info(f"on_click {event.xdata}, {event.ydata}")
                if traces:
                    traces.plot_trace_idx(cur_idx_plot)
                    traces.plot_ps_trace_idx(cur_idx_plot)
                    plt.show()
                plt.draw()

        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title(f"{self.name}\nDU network : {title}")
        vmin = a_values.min()
        vmax = a_values.max()
        norm_user = colors.LogNorm(vmin=vmin, vmax=vmax)
        if scale == "log":
            pass
        elif scale == "lin":
            norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            logger.error(f'scale must be in ["log","lin"]')
        scm = ax1.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=norm_user,
            s=size_circle,
            c=a_values,
            edgecolors="k",
            cmap="Reds",
        )
        fig.colorbar(scm)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        if traces:
            anch_du = AnchoredText("DU", prop=dict(size=10), frameon=False, loc="upper left")
            anch_val = AnchoredText(
                "Value max", prop=dict(size=10), frameon=False, loc="upper right"
            )
            ax1.add_artist(anch_du)
            ax1.add_artist(anch_val)
            plt.connect("button_press_event", on_click)
            plt.connect("motion_notify_event", on_move)

    def plot_footprint_4d(self, o_tr, title=""):  # pragma: no cover
        """
        Plot footprint of time max by DU and value max by component

        :param o_tr: object traces
        :type o_tr: Handling3dTracesOfEvent
        :param title: title of plot
        :type title: str
        """

        def subplot(plt_axis, a_values, traces=None, cpnt="", scale="log"):
            ax1 = plt_axis
            size_circle = 80
            cur_idx_plot = -1

            ax1.set_title(cpnt)
            if type(scale) is str:
                my_cmaps = "Blues"
                vmin = a_values.min()
                vmax = a_values.max()
                norm_user = colors.LogNorm(vmin=vmin, vmax=vmax)
                if scale == "log":
                    pass
                elif scale == "lin":
                    norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    logger.error(f'scale must be in ["log","lin"]')
            else:
                print("Use scale as norm")
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
            # plt.ylabel("[m]")
            # plt.xlabel("[m]")
            return scm

        fig, ax = plt.subplots(2, 2)

        t_max, _ = o_tr.get_tmax_vmax()
        v_max = np.max(np.abs(o_tr.traces), axis=2)
        ret_scat = subplot(ax[0, 0], t_max, cpnt="Time of max value", scale="lin")
        fig.colorbar(ret_scat)
        # same scale for
        vmin = v_max.min()
        vmax = v_max.max()
        norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        ret_scat = subplot(ax[1, 0], v_max[:, 0], o_tr, f"{o_tr.axis_name[0]}", norm_user)
        fig.colorbar(ret_scat)
        ret_scat = subplot(ax[0, 1], v_max[:, 1], o_tr, f"{o_tr.axis_name[1]}", norm_user)
        fig.colorbar(ret_scat)
        ret_scat = subplot(ax[1, 1], v_max[:, 2], o_tr, f"{o_tr.axis_name[2]}", norm_user)
        fig.colorbar(ret_scat)

    def plot_footprint_time(self, a_time, a3_values, title=""):  # pragma: no cover
        """
        Interactive plot, footprint max value for time defined by user with slider widget

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
        fig.colorbar(scat)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
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
