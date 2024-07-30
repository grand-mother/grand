'''
Created on Oct 5, 2022

@author: root
'''
'''
Created on 5 oct. 2022

@author: jcolley
'''
import argparse
from grand.io.root_file import FileSimuEfield
from grand.basis.traces_event import Handling3dTracesOfEvent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as colors

def plot_trace_animate(self, a_time, a3_values, title="", size_circle=200):
    from matplotlib.animation import FuncAnimation
    import matplotlib

    # same number of sample
    assert a_time.shape[0] == a3_values.shape[2]
    # we plot norm of 3D vector
    a_norm_val = np.linalg.norm(a3_values, axis=1)
    col_log = colors.LogNorm(clip=True)
    cmap = matplotlib.cm.get_cmap("Blues")
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    scat = ax.scatter(
        self.du_pos[:, 0],
        self.du_pos[:, 1],
        norm=col_log,
        s=size_circle,
        c=a_norm_val[:, 0],
        edgecolors="k",
        cmap="Blues",
    )
    fig.colorbar(scat)

    def update_time(frame_number):
        # 1) Can't update partial of scatter, like set_offsets, set_color, ...
        # so ... redraw all
        # 2) Can't clear before with : scatterm.clear()
        # idx_t = int((t_slider - a_time[0]) / delta_t)
        idx_t = frame_number
        print(idx_t)
        # new_col = cmap(col_log(a_norm_val[:,frame_number ]))
        # print(a_norm_val[:10,frame_number])
        # print(new_col[:10])
        scat.set_array(a_norm_val[:, frame_number])
        return scat
        # scat.set_color(new_col)
        # scat.set_edgecolors("k")
        # scat.set_offsets(self.du_pos)
        # fig.canvas.draw_idle()
        # print(t_slider, idx_t)

    #     scat = ax.scatter(
    #     self.du_pos[:, 0],
    #     self.du_pos[:, 1],
    #     #norm=col_log,
    #     s=size_circle,
    #     c=a_norm_val[:, idx_t],
    #     edgecolors="k",
    #     cmap="Blues",
    # )

    animation = FuncAnimation(
        fig, update_time, frames=range(0, a_time.shape[0], 80), interval=100
    )
    plt.show()

def animate():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    
    
    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])
    
    # Create rain data
    n_drops = 50
    rain_drops = np.zeros(n_drops, dtype=[('position', float, (2,)),
                                          ('size',     float),
                                          ('growth',   float),
                                          ('color',    float, (4,))])
    
    # Initialize the raindrops in random positions and with
    # random growth rates.
    rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
    rain_drops['growth'] = np.random.uniform(50, 200, n_drops)
    
    # Construct the scatter which we will update during animation
    # as the raindrops develop.
    scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
                      s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                      facecolors='none')
    
    
    def update(frame_number):
        # Get an index which we can use to re-spawn the oldest raindrop.
        current_index = frame_number % n_drops
    
        # Make all colors more transparent as time progresses.
        rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
        rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)
    
        # Make all circles bigger.
        rain_drops['size'] += rain_drops['growth']
    
        # Pick a new position for oldest rain drop, resetting its size,
        # color and growth factor.
        rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
        rain_drops['size'][current_index] = 5
        rain_drops['color'][current_index] = (0, 0, 0, 1)
        rain_drops['growth'][current_index] = np.random.uniform(50, 200)
    
        # Update the scatter collection, with the new colors, sizes and positions.
        scat.set_edgecolors(rain_drops['color'])
        scat.set_sizes(rain_drops['size'])
        scat.set_offsets(rain_drops['position'])
    
    
    # Construct the animation, using the update function as the animation director.
    animation = FuncAnimation(fig, update, interval=10)
    plt.show()

def other():
    pass
    # def on_move(eve=nt):
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

def example():              
            
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
    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    
    
    def reset(event):
        freq_slider.reset()
        #amp_slider.reset()
    button.on_clicked(reset)
    
    plt.show()
    
    
def animate_2():
    import itertools
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    
    def data_gen():
        for cnt in itertools.count():
            t = cnt / 10
            yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)
    
    
    def init():
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []
    
    
    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()
    
        if t >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()
        line.set_data(xdata, ydata)
    
        return line,
    
    ani = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
    plt.show()

    
if __name__ == '__main__':
    animate_2()
    
    
