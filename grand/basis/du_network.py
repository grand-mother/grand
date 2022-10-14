"""

"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider

# from matplotlib.backend_bases import MouseButton

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

    ### PLOT

    def plot_du_pos(self):
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_values(self, a_values, title="", size_circle=200): # pragma: no cover
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.backend_bases import MouseButton

        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            return np.argmin(dist_2)

        def on_click(event):
            if event.button is MouseButton.LEFT:
                # print(f'data coords {event.xdata} {event.ydata},',f'pixel coords {event.x} {event.y}')
                idx = closest_node(np.array([event.xdata, event.ydata]), self.du_pos[:, :2])
                atl = AnchoredText(
                    f"{self.du_id[idx]}", prop=dict(size=10), frameon=True, loc="upper left"
                )
                atl.patch.set_boxstyle("Circle, pad=0.3")
                atr = AnchoredText(
                    f"{a_values[idx]:.2e}", prop=dict(size=10), frameon=True, loc="upper right"
                )
                atr.patch.set_boxstyle("Square, pad=0.3")
                ax.add_artist(atl)
                ax.add_artist(atr)
                plt.draw()

        vmin = a_values.min()
        vmax = a_values.max()
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"{self.name}\nDU network : {title}")
        scm = ax.scatter(
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
        ax.add_artist(atl)
        ax.add_artist(atr)
        plt.connect("button_press_event", on_click)

    def plot_trace_time(self, a_time, a3_values, title="", size_circle=200): # pragma: no cover
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.backend_bases import MouseButton
        import matplotlib

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
        idx_mean = int(a_time.shape[0] // 2)
        # image
        # plt.figure()
        # print(a_norm_val.shape)
        # plt.imshow(a_norm_val, norm=col_log)
        # for update function
        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            return np.argmin(dist_2)

        def on_click(event):
            if event.button is MouseButton.LEFT:
                # print(f'data coords {event.xdata} {event.ydata},',f'pixel coords {event.x} {event.y}')
                idx = closest_node(np.array([event.xdata, event.ydata]), self.du_pos[:, :2])
                atl = AnchoredText(
                    f"{self.du_id[idx]}", prop=dict(size=10), frameon=True, loc="upper left"
                )
                atl.patch.set_boxstyle("Circle, pad=0.3")
                atr = AnchoredText(
                    f"{a_norm_val[idx, g_idx]}", prop=dict(size=10), frameon=True, loc="upper right"
                )
                atr.patch.set_boxstyle("Square, pad=0.3")
                ax.add_artist(atl)
                ax.add_artist(atr)
                plt.draw()

        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        scat = ax.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=col_log,
            s=size_circle,
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
        print(a_time[0], a_time[-1], a_time[100])
        time_slider = Slider(
            ax=axe_idx,
            label="time ns:",
            valmin=float(a_time[0]),
            valmax=float(a_time[-1]),  # a_time.shape[0]-1,
            valinit=float(a_time[0]),
        )
        # The function to be called anytime a slider's value changes
        def update_time(t_slider):
            # 1) Can't update partial of scatter, like set_offsets, set_color, ...
            # so ... redraw all
            # 2) Can't clear before with : scatterm.clear()
            # idx_t = int((t_slider - a_time[0]) / delta_t)
            idx_t = t_slider
            # print(t_slider, type(t_slider))
            # print(t_slider, idx_t)
            # ax.scatter(
            #     self.du_pos[:, 0],
            #     self.du_pos[:, 1],
            #     norm=col_log,
            #     s=size_circle,
            #     c=a_norm_val[:, int(idx_t)],
            #     edgecolors="k",
            #     cmap="Blues",
            # )
            frame_number = int((t_slider - a_time[0]) / delta_t)
            # print(t_slider, frame_number)
            # new_col = cmap_b(col_log(a_norm_val[:,frame_number ]))
            # print(a_norm_val[:10,frame_number])
            # print(new_col[:10])
            # scat.set_color(new_col)
            scat.set_array(a_norm_val[:, frame_number])
            # scat.set_edgecolors("k")
            # scat.set_offsets(self.du_pos)
            # fig.canvas.draw_idle()
            # fig.canvas.draw_idle()
            # idx = (closest_node(np.array([event.xdata,event.ydata]), self.du_pos[:,:2]))
            # atr = AnchoredText(f"{a_norm_val[idx, idx_t]}", prop=dict(size=10), frameon=True, loc='upper right')
            # atr.patch.set_boxstyle("Square, pad=0.3")
            # ax.add_artist(atr)
            # plt.draw()
            return scat

        time_slider.on_changed(update_time)
        # plt.connect("button_press_event", on_click)
        # WARNING: we must used plt.show() of this module not in another module
        #          else slider is blocked
        plt.show()
