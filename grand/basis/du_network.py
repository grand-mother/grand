"""

"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider
#from matplotlib.backend_bases import MouseButton

logger = getLogger(__name__)

class DetectorUnitNetwork:
    """
    classdocs
    """

    def __init__(self, name="NotDefined"):
        """
        Constructor
        """
        self.name = name

    def init_pos_id(self, du_pos, du_id):
        self.du_pos = du_pos
        self.du_id = du_id
        assert isinstance(self.du_pos, np.ndarray)
        assert isinstance(self.du_id, np.ndarray)
        assert du_pos.shape[0] == du_id.shape[0]
        assert du_pos.shape[1] == 3

    def reduce_nb_du(self, new_nb_du):
        """
        feature to reduce computation and debugging
        :param new_nb_du:
        """
        self.du_id = self.du_id[:new_nb_du]
        self.du_pos = self.du_pos[:new_nb_du, :]

    def get_sub_network(self, l_id):
        sub_net = DetectorUnitNetwork("sub-network of " + self.name)
        sub_net.init_pos_id(self.du_pos[l_id], self.du_id[l_id])
        return sub_net

    def get_pos_id(self, l_id):
        pass

    def get_surface(self):
        pass

    def get_max_dist_du(self, l_id):
        pass

    def plot_du_pos(self):
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_value(self, values, title="", size_circle=100):
        from matplotlib.offsetbox import AnchoredText
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"{self.name}\nDU network : {title}")
        scm = ax.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=colors.LogNorm(vmin=values.min(), vmax=values.max()),
            s=size_circle,
            c=values,
            edgecolors="k",
            cmap="Reds",
        )
        fig.colorbar(scm)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        atl = AnchoredText("24", prop=dict(size=10), frameon=True, loc='upper left')
        atl.patch.set_boxstyle("Circle, pad=0.3")
        atr = AnchoredText("1.23e4", prop=dict(size=10), frameon=True, loc='upper right')
        atr.patch.set_boxstyle("Square, pad=0.3")
        ax.add_artist(atl)
        ax.add_artist(atr)
        
        
    def plot_trace_time(self, a_time, a_values, title="", size_circle=100):
        assert a_time.shape[0] == a_values.shape[1]
        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        idx_mean = int(a_time.shape[0] // 2)
        val_min = a_values.min()
        if val_min <= 0:
            val_min = 0.1
        col_log = colors.LogNorm(vmin=val_min, vmax=a_values.max())
        scatterm = ax.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=col_log,
            s=size_circle,
            c=a_values[:, idx_mean],
            edgecolors="k",
            cmap="Reds",
        )
        fig.colorbar(scatterm)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        fig.subplots_adjust(left=0.25, bottom=0.25)
        # Make a horizontal slider to control the frequency.
        axe_idx = fig.add_axes([0.2, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax=axe_idx,
            label="idx",
            valmin=0.1,
            valmax=a_time.shape[0],
            valinit=idx_mean,
        )
        # The function to be called anytime a slider's value changes
        def update_time(idx_t):
            # 1) Can't update partial of scatter, like set_offsets, set_color, ...
            # so ... redraw all
            # 2) Can't clear before with : scatterm.clear()
            scatterm = ax.scatter(
                self.du_pos[:, 0],
                self.du_pos[:, 1],
                norm=col_log,
                s=size_circle,
                c=a_values[:, int(idx_t)],
                edgecolors="k",
                cmap="Reds",
            )
            fig.canvas.draw_idle()
        scatterm = time_slider.on_changed(update_time)
        # WARNING: we must used plt.show() of this module not in another module 
        #          else slider is blocked              
        plt.show()
        