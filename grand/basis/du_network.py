"""
Handling DU network, place for footprint plot
"""
from logging import getLogger

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider
from matplotlib.offsetbox import AnchoredText
from matplotlib.backend_bases import MouseButton


logger = getLogger(__name__)


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
        du_pos: float (N,3) position of DU
        du_id: float (N) identifier of DU
        """
        self.du_pos = du_pos
        self.du_id = du_id
        assert isinstance(self.du_pos, np.ndarray)
        assert isinstance(self.du_id, np.ndarray)
        assert du_pos.shape[0] == du_id.shape[0]
        assert du_pos.shape[1] == 3

    def reduce_nb_du(self, new_nb_du):
        """
        Fature to reduce computation and debugging
        new_nb_du: integer
        """
        self.du_id = self.du_id[:new_nb_du]
        self.du_pos = self.du_pos[:new_nb_du, :]

    def get_sub_network(self, l_id):
        """

        :param l_id:
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
        """
        # TODO:
        raise NotImplementedError

    def get_max_dist_du(self, l_id):
        """

        :param l_id:
        :type l_id:
        """
        # TODO:
        raise NotImplementedError

    ### PLOT

    def plot_du_pos(self):  # pragma: no cover
        """
        Figure with DU position
        """
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_footprint_1d(self, a_values, title="", traces=None):  # pragma: no cover
        """
        Interactive footprint double click on DU draw trace associated and power spectrum

        Footprint of ma
        :param a_values: intensity associated to DU
        :type a_values: float (nb DU)
        :param title: title of figure
        :type title: string
        :param traces: object traces
        :type traces: Handling3dTracesOfEvent
        """
        size_circle = 200

        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            return np.argmin(dist_2)

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                logger.info(f"on_click {event.xdata}, {event.ydata}")
                idx = closest_node(np.array([event.xdata, event.ydata]), self.du_pos[:, :2])
                atl = AnchoredText(
                    f"{self.du_id[idx]}", prop=dict(size=10), frameon=True, loc="upper left"
                )
                atl.patch.set_boxstyle("Circle, pad=0.3")
                atr = AnchoredText(
                    f"{a_values[idx]:.2e}", prop=dict(size=10), frameon=True, loc="upper right"
                )
                atr.patch.set_boxstyle("Square, pad=0.3")
                ax1.add_artist(atl)
                ax1.add_artist(atr)
                if traces:
                    traces.plot_trace_idx(idx)
                    traces.plot_ps_trace_idx(idx)
                    plt.show()
                plt.draw()

        vmin = a_values.min()
        vmax = a_values.max()
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_title(f"{self.name}\nDU network : {title}")
        scm = ax1.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            s=size_circle,
            c=a_values,
            edgecolors="k",
            cmap="Reds",
        )
        fig.colorbar(scm)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        atl = AnchoredText("DU", prop=dict(size=10), frameon=True, loc="upper left")
        atl.patch.set_boxstyle("Circle, pad=0.3")
        atr = AnchoredText("value DU", prop=dict(size=10), frameon=True, loc="upper right")
        atr.patch.set_boxstyle("Square, pad=0.3")
        ax1.add_artist(atl)
        ax1.add_artist(atr)
        plt.connect("button_press_event", on_click)

    def plot_footprint_time(self, a_time, a3_values, title=""):  # pragma: no cover
        """

        :param a_time:
        :type a_time:
        :param a3_values:
        :type a3_values:
        :param title:
        :type title:
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
