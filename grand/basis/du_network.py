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
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"{self.name}\nDU network\n{title}")
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

    def plot_trace_time(self, a_time, a_values, title="", size_circle=100):
        assert a_time.shape[0] == a_values.shape[1]
        print("aaaa")
        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        idx_t = a_time.shape[0] // 2
        print(idx_t)
        print(a_time.shape[0])
        print(a_values.min(), a_values.max())
        col_log = colors.LogNorm(vmin=a_values.min(), vmax=a_values.max())
        scatterm = ax.scatter(
            self.du_pos[:, 0],
            self.du_pos[:, 1],
            norm=col_log,
            s=size_circle,
            c=a_values[:, idx_t],
            edgecolors="k",
            cmap="Reds",
        )
        fig.colorbar(scatterm)
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        fig.subplots_adjust(left=0.25, bottom=0.25)
        # Make a horizontal slider to control the frequency.
        # axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        axe_idx = fig.add_axes([0.2, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax=axe_idx,
            label="idx",
            valmin=0.1,
            valmax=998.0,
            valinit=123.0,
        )
        # The function to be called anytime a slider's value changes
        def update_time(idx_t):
            #global scatterm
            print(f"update {idx_t}")            
            #scatterm.clear()
            scatterm = ax.scatter(
                self.du_pos[:, 0],
                self.du_pos[:, 1],
                norm=col_log,
                s=size_circle,
                c=a_values[:, int(idx_t)],
                edgecolors="k",
                cmap="Reds",
            )
            # scm.set_offsets(np.stack(
            #     self.du_pos[:, 0],
            #     self.du_pos[:, 1]), axis=1
            # )
            # scm.set_norm(col_log)
            # scm.set_sizes(size_circle)
            # scm.set_color(c=a_values[:, int(time_slider.val)])
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time)
        # def on_move(event):
        #     if event.inaxes:
        #         print(f'data coords {event.xdata} {event.ydata},',
        #               f'pixel coords {event.x} {event.y}')
        #
        #
        # def on_click(event):
        #     if event.button is MouseButton.LEFT:
        #         #print('disconnecting callback')
        #         #plt.disconnect(binding_id)
        #         print(f'data coords {event.xdata} {event.ydata},',
        #               f'pixel coords {event.x} {event.y}')
        #
        # #binding_id = plt.connect('motion_notify_event', on_move)
        # plt.connect('button_press_event', on_click)        
        plt.show()
        
    def plot_trace_time_test(self, a,b,c):
        # The parametrized function to be plotted
        def f(t, amplitude, frequency):
            return amplitude * np.sin(2 * np.pi * frequency * t)
        
        t = np.linspace(0, 1, 1000)
        
        # Define initial parameters
        init_amplitude = 5
        init_frequency = 3
        
        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
        ax.set_xlabel('Time [s]')
        
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.25, bottom=0.25)
        
        # Make a horizontal slider to control the frequency.
        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        freq_slider = Slider(
            ax=axfreq,
            label='Frequency [Hz]',
            valmin=0.1,
            valmax=30,
            valinit=init_frequency,
        )
        
        # Make a vertically oriented slider to control the amplitude
        # axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        # amp_slider = Slider(
        #     ax=axamp,
        #     label="Amplitude",
        #     valmin=0,
        #     valmax=10,
        #     valinit=init_amplitude,
        #     orientation="vertical"
        # )
        
        
        # The function to be called anytime a slider's value changes
        def update(val):
            line.set_ydata(f(t, init_amplitude, freq_slider.val))
            fig.canvas.draw_idle()
        
        
        # register the update function with each slider
        freq_slider.on_changed(update)
        #amp_slider.on_changed(update)
        plt.show()
