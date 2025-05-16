"""
MIT License.

Copyright (c) 2021 GRAND Collaboration
contact: rkoirala@nju.edu.cn
contact2 (update to root): claire.guepin@lupm.in2p3.fr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import numpy as np
import pandas as pd
# http://holoviews.org/getting_started/index.html
import panel as pn
import holoviews as hv
from holoviews import opts, dim
from astropy.time import Time

from scipy.signal import hilbert
import scipy.interpolate as scipolate
import mix  # functions written by Valentin Decoene.
import seaborn as sns  # used for color pallettes.

from grand.aoi import EventList


class EventViewer:
    """
    EventViewer for GRAND events.

    Using "holoviews" software as a base for visualization and creating widget.
    Needs python 3.6 or higher.
    $ pip3 install "holoviews[all]"
    holoview should install bokeh, if not install it using pip3
    $ pip3 install bokeh
    """

    def __init__(self):
        print("Initialization")
        self.geofile = geofile
        self.datadir = datadir
        # Time step to look for antennae begin hit
        self.tstep = 1500
        # Minimum radio frequency in hertz
        self.fmin = 50.e6
        # Maximum radio frequency in hertz
        self.fmax = 200.e6
        # Show the shower core
        self.plt_core = True

    def get_geometry(self):
        """Get layout of proposed geometry of GP300."""
        self.geo_df = pd.read_csv(self.geofile, sep=" ", usecols=[1, 2, 3, 4])
        # sometimes int are used as ID.
        self.geo_df['ID'] = np.array([str(ant_name)
                                     for ant_name in self.geo_df['ID']])
        # x-coordinate of all antenna in km.
        self.posx = self.geo_df['X']/1.e3
        # y-coordinate of all antenna in km.
        self.posy = self.geo_df['Y']/1.e3
        '''
        For 3D:
        self.posz = self.geo_df['Z']/1.e3  # z-coordinate of all antenna in km.
        '''

    def get_data(self):
        """Collect event information.

        Required to make plots in event-display.
        This method is called everytime a new data file is given as an input.
        """
        print("Reading directory", self.datadir)

        # Create the EventList with the specified file
        el = EventList(self.datadir)
        print("Number of events: %i" % el.get_number_of_events())

        # Iterate through some events
        for i in range(862, 863):  # 862, 980, 994 GOOD EVENTS
            e = el.get_event(entry_number=i)
            print(f"Event {i}, "
                  + f"du_id {e.efields[0].du_id}, "
                  + f"time {e.efields[0].t0}")
            if e.tsimshower.energy_primary/1e9 > 1.:
                break

        # Info related to the shower.
        self.primary = e.tsimshower.primary_type
        self.energy = e.tsimshower.energy_primary/1e9  # GeV to EeV
        self.zenith = np.deg2rad(e.tsimshower.zenith)
        self.azimuth = np.deg2rad(e.tsimshower.azimuth)
        self.x_xmax, self.y_xmax, self.z_xmax = e.tsimshower.xmax_pos_shc
        self.slant_xmax = e.tsimshower.xmax_grams

        # Magnetic field
        Bfield = e.tsimshower.magnetic_field
        Bnorm = np.sqrt(np.sum(Bfield**2))
        print("Bfield norm = %.2e muT (unsure unit)" % Bnorm)
        self.bx = Bfield[0]/Bnorm
        self.by = Bfield[1]/Bnorm
        self.bz = Bfield[2]/Bnorm

        # Antenna postions
        self.antpos = np.array([[ant.position.x[0],
                                 ant.position.y[0],
                                 ant.position.z[0]] for ant in e.antennas])

        # Efield
        self.efields = e.efields
        # Filtered peak time and peak amplitude
        # Calculated using hilbert transform
        # =====================================================================
        # DUPLICATE, PUT THIS IN A FUNCTION
        # =====================================================================
        self.t0 = np.zeros(len(self.antpos))
        self.peaktime = np.zeros(len(self.antpos))
        self.peakamplitude = np.zeros(len(self.antpos))
        for i in range(len(self.antpos)):
            self.t0[i] = Time(e.efields[i].t0).jd
            efield_filt = mix.filters_root(e.efields[i].t_vector*1.e-9,
                                           e.efields[i].trace,
                                           FREQMIN=self.fmin,
                                           FREQMAX=self.fmax)
            hilbert_amp = np.abs(hilbert(efield_filt[1:4, :]))
            self.peakamplitude[i] = max([max(hilbert_amp[0, :]),
                                         max(hilbert_amp[1, :]),
                                         max(hilbert_amp[2, :])])
            self.peaktime[i] = efield_filt[0, np.where(
                hilbert_amp == self.peakamplitude[i])[1][0]]

        # =====================================================================
        # DOES NOT WORK YET
        # # index based on increasing time.
        # # sorted_indx = self.t0.argsort()
        # sorted_indx = self.peaktime.argsort()
        # print(sorted_indx)
        # self.antpos = self.antpos[sorted_indx]
        # self.peaktime = self.peaktime[sorted_indx]
        # self.peakamplitude = self.peakamplitude[sorted_indx]
        # for i in range(len(self.antpos)):
        #     self.efields[i] = self.efields[sorted_indx[i]]
        # =====================================================================

        # Positions of hit antennae in GRAND coordinate system (cs)
        self.hitX = self.antpos[:, 0]
        self.hitY = self.antpos[:, 1]
        self.hitZ = self.antpos[:, 2]
        # Position of shower core in GRAND cs
        self.corex = e.tsimshower.shower_core_pos[0]
        self.corey = e.tsimshower.shower_core_pos[1]
        # With respect to shower core
        self.hitXc = self.hitX-self.corex
        self.hitYc = self.hitY-self.corey
        self.hitZc = self.hitZ

        # Additional info
        self.palette_color = self.select_color()
        # Time boundary to look for hits
        self.tbins = np.arange(min(self.peaktime)-2*self.tstep,
                               max(self.peaktime)+2*self.tstep, self.tstep)
        self.nhits = len(self.hitX)

    def get_trace(self):
        """Get time traces of data."""
        lw = 1.5  # line-width of traces curves.
        alp = 0.9  # alpha on used color.

        tcurve = {}
        tcurve_h = {}

        for i in range(len(self.antpos)):
            efield_trace_loc = self.efields[i]
            ant_id = str(efield_trace_loc.du_id)
            efield = efield_trace_loc.trace
            efield_filt = mix.filters_root(efield_trace_loc.t_vector*1.e-9,
                                           efield_trace_loc.trace,
                                           FREQMIN=self.fmin,
                                           FREQMAX=self.fmax)
            hilbert_amp = np.abs(hilbert(efield_filt[1:4, :]))

            # plot traces
            curvex = hv.Curve(
                efield[0, :], 'Time Bins', 'E-field Trace', label='Ex')\
                .opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp,
                      color='r')
            curvey = hv.Curve(
                efield[1, :], 'Time Bins', 'E-field Trace', label='Ey')\
                .opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp,
                      color='steelblue')
            curvez = hv.Curve(
                efield[2, :], 'Time Bins', 'E-field Trace', label='Ez')\
                .opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp,
                      color='olive')
            curve = curvex*curvey*curvez

            ymin = min([min(efield[0, :]), min(
                efield[1, :]), min(efield[2, :])])
            ymin = ymin - .05*abs(ymin)
            ymax = max([max(efield[0, :]), max(
                efield[1, :]), max(efield[2, :])])
            ymax = ymax + .05*abs(ymax)

            curve.opts(show_grid=True, title='Antenna: '+ant_id,
                       toolbar='above',
                       xlim=(-1, len(efield[0, :])+1),
                       # important to dynamically change ylim of traces.
                       ylim=(ymin, ymax),
                       legend_position='top_right',
                       legend_cols=3)
            tcurve[i] = curve

            # plot hilbert transform
            curvexh = hv.Curve(
                hilbert_amp[0, :], 'Time Bins', 'E-field [μV/m]')\
                .opts(line_width=lw, tools=['hover'], alpha=alp-0.1,
                      color='r')
            curveyh = hv.Curve(
                hilbert_amp[1, :], 'Time Bins', 'E-field [μV/m]')\
                .opts(line_width=lw, tools=['hover'], alpha=alp,
                      color='steelblue')
            curvezh = hv.Curve(
                hilbert_amp[2, :], 'Time Bins', 'E-field [μV/m]')\
                .opts(line_width=lw, tools=['hover'], alpha=alp,
                      color='olive')
            curve_h = curvexh*curveyh*curvezh

            ymin_h = min([min(hilbert_amp[0, :]), min(
                hilbert_amp[1, :]), min(hilbert_amp[2, :])])
            ymax_h = max([max(hilbert_amp[0, :]), max(
                hilbert_amp[1, :]), max(hilbert_amp[2, :])])
            ymin_h = ymin_h - .05*abs(ymax_h)  # ymin is always 0.
            ymax_h = ymax_h + .05*abs(ymax_h)

            curve_h.opts(title='Hilbert Envelope',
                         show_grid=True,
                         xlim=(-1, len(hilbert_amp[0, :])+1),
                         ylim=(ymin_h, ymax_h),
                         toolbar='above')  # title='Hilbert Envelope',
            tcurve_h[i] = curve_h

        # to prevent from taking log on <1 numbers. Use any number >=1.
        self.Eweight = self.peakamplitude + 15.
        self.trace_collection = tcurve
        self.hilbert_collection = tcurve_h

    def pick_trace(self, index):
        """Pick Efield traces.

        Electric field traces from all hit antennae
        are collected in 'trace_collection'.
        This function picks trace from collected traces for a particular
        antenna for plotting when clicked on that antenna.
        Traces are plotted only for antennae that are hit.
        """
        if not index:
            c1 = hv.Curve([], 'Time Bins', 'E-field Trace')
            c2 = hv.Curve([], 'Time Bins', 'E-field Trace')
            c3 = hv.Curve([], 'Time Bins', 'E-field Trace')
            curve = c1*c2*c3
            return curve

        # index here is a list with 1 entry.
        antEtrace = self.trace_collection[index[0]]
        antEtrace.opts(width=side_width, height=side_height, show_grid=True,
                       fontsize={'title': 16,
                                 'labels': 13,
                                 'legend': 8,
                                 'xticks': 10,
                                 'yticks': 10})

        return antEtrace

    def pick_hilbert_trace(self, index):
        """Pick Hilbert envelops.

        Hilbert envelop of traces from all hit antennae
        are collected in 'hilbert_collection'.
        This function picks trace from collected traces for a particular
        antenna for plotting when clicked on that antenna.
        Traces are plotted only for antennae that are hit.
        """
        if not index:
            c1 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
            c2 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
            c3 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
            curve = c1*c2*c3
            return curve

        # index here is a list with 1 entry.
        antEtrace_h = self.hilbert_collection[index[0]]
        antEtrace_h.opts(width=side_width, height=side_height, show_grid=True,
                         fontsize={'title': 10,
                                   'labels': 13,
                                   'legend': 8,
                                   'xticks': 10,
                                   'yticks': 10})

        return antEtrace_h

    def plot_text(self, data=[]):
        """Print basic shower information on the display.

        To Do: Extend this to include experimental events.
        """
        quantity = ['Particle', 'Ene [EeV]', 'Zen [deg]',
                    'Azi [deg]', 'Xmax [g/cm2]']
        value = [self.primary,
                 round(self.energy, 2),
                 round(np.rad2deg(self.zenith), 2),
                 round(np.rad2deg(self.azimuth), 2),
                 round(self.slant_xmax, 2)]

        text = {'Quantity': quantity, 'Value': value}
        df = pd.DataFrame(text, columns=['Quantity', 'Value'])
        txt_table = hv.Table(df)
        txt_table.opts(height=200, width=250)

        return txt_table

    def peak_amplitude_ground_plane(self, data):
        """Plot interpolated peak amplitude in ground plane."""
        X, Y = np.meshgrid(np.linspace(self.hitX.min(), self.hitY.max(), 200),
                           np.linspace(self.hitY.min(), self.hitY.max(), 200))
        inter_peakamp_grd = scipolate.Rbf(
            self.hitX, self.hitY, self.peakamplitude,
            function='linear', epsilon=9)(X, Y)
        kdims = ['x_gp', 'y_gp']
        vdims = ['peakA']
        bounds = (self.hitX.min()/1.e3, self.hitY.min()/1.e3,
                  self.hitX.max()/1.e3, self.hitY.max()/1.e3)
        # np.flipud(data) is performed inside Image.
        # So it is done here to undo that process.
        # If not done, the image will be upside down.
        plot_cc = hv.Image(np.flipud(inter_peakamp_grd), kdims=kdims,
                           vdims=vdims, bounds=bounds)  \
            .opts(width=img_width,
                  height=img_height,
                  cmap='Spectral_r',
                  title='Ground Plane [km]',
                  xlabel='',
                  ylabel='',
                  tools=['hover'],
                  toolbar='below',
                  fontsize={'title': 10,
                            'labels': 11,
                            'xticks': 10,
                            'yticks': 10})
        return plot_cc

    def peak_amplitude_shower_plane(self, data):
        """Collect event info to make plots in event-display.

        This part of code came from Valentin Decoene.
        'data' is kept here only for syntax reason and is not used.
        Remove this in future.
        """
        self.k_shower = -np.array([np.sin(self.zenith)*np.cos(self.azimuth),
                                   np.sin(self.zenith)*np.sin(self.azimuth),
                                   np.cos(self.zenith)])
        # Position of antannae in shower coordinate system.
        self.x_sp, self.y_sp, self.z_sp = mix.get_in_shower_plane_root(
            np.array([self.hitXc, self.hitYc, self.hitZc]),
            self.k_shower,
            # z-value is not on the ground.
            np.array([0, 0, np.mean(self.hitZ)]),
            self.bx,
            self.by,
            self.bz
            )

        Xsp, Ysp = np.meshgrid(np.linspace(self.x_sp.min(),
                                           self.x_sp.max(),
                                           200),
                               np.linspace(self.y_sp.min(),
                                           self.y_sp.max(),
                                           200))
        inter_peakamp = scipolate.Rbf(self.x_sp,
                                      self.y_sp,
                                      self.peakamplitude,
                                      function='thin_plate',
                                      epsilon=9)(Xsp, Ysp)

        kdims = ['x_sp', 'y_sp']
        vdims = ['peakA']
        xmin = self.x_sp.min()/1.e3
        xmax = self.x_sp.max()/1.e3
        ymin = self.y_sp.min()/1.e3
        ymax = self.y_sp.max()/1.e3
        bounds = (xmin, ymin, xmax, ymax)
        lmax = max([xmax-xmin, ymax-ymin])

        # np.flipud(data) is performed inside hv.Image.
        # So it is done here to undo that process.
        # If not done, the image will be upside down.
        plot_cc = hv.Image(np.flipud(inter_peakamp), kdims=kdims, vdims=vdims,
                           bounds=bounds) \
            .opts(width=img_width,
                  height=img_height,
                  cmap='Spectral_r',
                  title='Shower Plane',
                  xlabel='vxB [km]',
                  ylabel='vx(vxB) [km]',
                  # this is for equal aspect ratio.
                  xlim=(xmin-0.005*lmax, xmin+1.005*lmax),
                  # this is for equal aspect ratio.
                  ylim=(ymin-0.005*lmax, ymin+1.005*lmax),
                  tools=['hover'],
                  toolbar='below',
                  fontsize={'title': 10,
                            'labels': 11,
                            'xticks': 10,
                            'yticks': 10})
        return plot_cc

    def peak_amplitude_angular_plane(self, data):
        """Plot cerenkov ring in angular plane.

        'data' is kept here only for syntax reason and is not used.
        Remove this in future.
        Note: calculations done here is borrowed from Valentin Decoene.
        """
        XmaxA_x = self.hitXc - self.x_xmax
        XmaxA_y = self.hitYc - self.y_xmax
        XmaxA_z = self.hitZc - self.z_xmax

        obs = np.array([XmaxA_x, XmaxA_y, XmaxA_z])
        ll = np.sqrt(XmaxA_x**2 + XmaxA_y**2 + XmaxA_z**2)
        u_ant = obs / ll
        cosw = np.dot(self.k_shower, u_ant)
        # sometimes value of cosine is 1.00000006 instead of 1.
        cosw[np.where(cosw > 1.)] = 1.

        self.w = np.arccos(cosw)
        # arctan2 chooses the quadrant properly.
        eta = np.arctan2(self.y_sp, self.x_sp)

        # self.x_angular = np.rad2deg(w)*np.sign(self.x_sp)
        # self.y_angular = np.rad2deg(w)*np.sign(self.y_sp)
        self.x_angular = np.rad2deg(self.w)*np.cos(eta)
        self.y_angular = np.rad2deg(self.w)*np.sin(eta)

        Xang, Yang = np.meshgrid(np.linspace(self.x_angular.min(),
                                             self.x_angular.max(),
                                             200),
                                 np.linspace(self.y_angular.min(),
                                             self.y_angular.max(),
                                             200))
        inter_peakamp = scipolate.Rbf(self.x_angular,
                                      self.y_angular,
                                      self.peakamplitude,
                                      function='linear', epsilon=9)(Xang, Yang)

        kdims = ['x_ap', 'y_ap']
        vdims = ['peakA']
        xmin = self.x_angular.min()
        xmax = self.x_angular.max()
        ymin = self.y_angular.min()
        ymax = self.y_angular.max()
        bounds = (xmin, ymin, xmax, ymax)
        lmax = max([xmax-xmin, ymax-ymin])
        bounds = (xmin, ymin, xmax, ymax)
        lmax = max([self.x_angular.max()-self.x_angular.min(),
                   self.y_angular.max()-self.y_angular.min()])
        # np.flipud(data) is performed inside Image.
        # So it is done here to undo that process.
        # If not done, the image will be upside down.
        plot_cc = hv.Image(np.flipud(inter_peakamp), kdims=kdims, vdims=vdims,
                           bounds=bounds)   \
            .opts(width=img_width,
                  height=img_height,
                  cmap='Spectral_r',
                  title='Angular Plane',
                  xlabel='ω along vxB [deg]',
                  ylabel='ω along vx(vxB) [deg]',
                  # this is for equal aspect ratio.
                  xlim=(xmin-0.005*lmax, xmin+1.005*lmax),
                  # this is for equal aspect ratio.
                  ylim=(ymin-0.005*lmax, ymin+1.005*lmax),
                  tools=['hover'],
                  toolbar='below',
                  fontsize={'title': 10,
                            'labels': 11,
                            'xticks': 10,
                            'yticks': 10})
        return plot_cc

    def peak_cerenkov_angle(self, data):
        """Plot cerenkov ring in angular plane."""
        # 'data' is kept here only for syntax reason and is not used.
        # Remove this in future.
        kdims = ['Omega', 'peakA']
        x_omega = np.rad2deg(self.w)*np.sign(self.y_sp)
        plot_ca = hv.Points(np.column_stack((x_omega, self.peakamplitude)),
                            kdims=kdims) \
            .opts(
            # width=img_width, height=img_height,
            color='k',
            alpha=0.9,
            tools=['hover'],
            size=4)

        xmin = min(x_omega) - 0.1
        xmax = max(x_omega) + 0.1
        ymin = min(self.peakamplitude) * 1.05
        ymax = max(self.peakamplitude) * 1.05
        plot_ca.opts(
            legend_position='bottom_right',
            legend_cols=3,
            toolbar='below',
            show_grid=True,
            xlabel='ω [deg]',
            ylabel='Peak Amp [μV/m]',
            xlim=(xmin, xmax),
            ylim=(ymin, ymax))

        return plot_ca

    def animate(self, event):
        """Control Play button. Plot hits binned in time.

        THIS FUNCTION NEEDS TO BE UPDATED
        """
        if self.play_button.name == '▶ Play':
            self.play_button.name = '❚❚ Pause'
            # Check if the input file name has been changed.
            # If changed start plotting the new event
            # after 'Play' button is clicked.
            filename0 = self.input_file.filename
            try:
                if filename0 is not None:
                    findex0 = np.where(
                        '/' == np.array([i for i in filename0]))[0]
                    if len(findex0) != 0:
                        filename = datadir + filename0[findex0[-1]+1:]
                    else:
                        filename = datadir + filename0

                    if filename != self.hdffile:
                        # if new hdf file provided, start from the beginning.
                        self.hdffile = filename
                        # get hitX, hitY, ..., tbins for new input hdf file.
                        self.get_data()
                        # get electric field traces from new input hdf file.
                        self.get_trace()
                        # sending nothing, calling to replot with updated data.
                        self.stream_ring.send(data=[])

                    if self.choose_color.value != self.select_color():
                        self.get_data()
                        self.stream_ring.send(data=[])

                self.plt_core = True
                indx = 0
                # loop over all hits and send data via pipe to plot one by one.
                while indx < len(self.tbins):
                    # It is faster to plot and hit-evolution looks smooth
                    # if hits are binned in time steps.
                    mask = self.peaktime <= self.tbins[indx]
                    # select x-coordinate of hit antennae before a given time.
                    x = np.array(self.hitX)[mask]
                    # select y-coordinate of hit antennae before a given time.
                    y = np.array(self.hitY)[mask]
                    # select list of time before the boundary time.
                    t = np.array(self.peaktime)[mask]
                    # Weight based on peak amplitude.
                    # This is an adhoc weight and has no physical meaning.
                    wt = np.array(self.Eweight)[mask]
                    # select color from a palette that was created
                    # based on time of hit.
                    color = np.array(self.palette_color)[mask]
                    # tunnel hits info to a dynamic map.
                    self.stream_hits.send((x, y, t, wt, color))
                    indx += 1

                # Show play button after an event is displayed.
                self.play_button.name = '▶ Play'

            except FileNotFoundError:
                # After all hits are plotted, change 'Pause' button to 'Play'.
                print("ERROR: Choose a file to display event.")
                self.play_button.name = '▶ Play'

    def plot_hits(self, data):
        """Control evolution of hits on detector geometry.

        Color represents the time of hit and the size of circle represents
        the size of signal on antennae.
        But the size of circle and the strength of electric field/voltage
        on antennae are not directly related.
        Hits to be plotted are binned in time so that hits are evolved in a
        reasonable speed.
        Note that the provided antennae information (x,y,t) are already sorted
        based by time of hit.
        'data' is sent here from 'animate' function inside a while loop.
        """
        x = data[0]  # updated x-coordinate of hit tanks to be plotted.
        y = data[1]  # updated y-coordinate of hit tanks to be plotted.
        t = data[2]  # updated hit time of hit tanks to be plotted.
        wt = data[3]  # updated weight of hit tanks to be plotted.
        color = data[4]  # updated colors to represent time of hit.
        kdims = ['X', 'Y']
        vdims = ['Weight', 'Time', 'Color']

        # if empty data is sent, prevents code to fail and plots empty hits
        if len(x) == 0:
            fig = hv.Points([], kdims=kdims, vdims=vdims)
            fig.opts(opts.Points(width=main_width,
                     height=main_height, tools=['hover']))

        else:
            pd_data = pd.DataFrame(data={'X': x.astype(float)/1.e3,  # m to km
                                         'Y': y.astype(float)/1.e3,  # m to km
                                         'Weight': wt.astype(float),
                                         'Time': t.astype(float),
                                         'Color': color})

            minwt = np.abs(np.log10(min(pd_data['Weight'])))
            maxwt = np.abs(np.log10(max(pd_data['Weight'])))
            # create a holoview dataset from pandas dataframe.
            ds = hv.Dataset(pd_data)
            '''This is the part where hits are plotted.
            This function is called many times and number of
            hits are added in each call until all hits are included.'''
            fig = hv.Points(ds, kdims=kdims, vdims=vdims)
            '''Add options to the plot.'''
            fig.opts(opts.Points(width=main_width, height=main_height,
                                 # signal strenght. This is arbitrary.
                                 size=10*((np.abs(np.log10(dim('Weight')))
                                           - minwt)/(maxwt-minwt)+1),
                                 marker='circle',
                                 color='Color',
                                 alpha=0.95,
                                 tools=['hover']))
        return fig

    def plot_core(self, data):
        """Plot shower core."""
        if self.plt_core:
            return hv.Points((self.corex/1.e3, self.corey/1.e3)).opts(
                color='k', marker='star_dot', size=25)
        else:
            return hv.Points([]).opts(
                color='k', marker='star_dot', size=25)

    def select_color(self):
        """Select color."""
        self.color_pallete = sns.palettes.color_palette(
            self.choose_color.value, len(self.hitX)).as_hex()
        return self.color_pallete

    def view(self):
        """View figures.

        All necessary process are called and managed from here.
        Updating hits plot dynamically is done from here.
        """
        # =====================================================================
        # Choose color
        self.choose_color = pn.widgets.Select(
            options=color_options, value='RdBu_r')

        # =====================================================================
        # Browse event file to display
        self.input_file = pn.widgets.FileInput(accept='.hdf5, .root')
        # self.input_file.filename = self.hdffile
        # get updated position of antennae, (i.e. posx, posy)
        self.get_geometry()
        # get updated hitAnt, hitX, hitY, hitT etc...
        self.get_data()
        # get updated electric field traces and hilbert envelop
        self.get_trace()

        # =====================================================================
        # Plot detector geometry with all antennae position
        antpos = np.column_stack((self.posx, self.posy))
        # .opts(fontsize={'xticks': 10, 'yticks': 10})
        antposplot = hv.Points(antpos, kdims=['X', 'Y'])
        antposplot.opts(opts.Points(width=main_width,
                                    height=main_height,
                                    marker='circle',
                                    size=8,
                                    tools=['hover'],
                                    xlabel='South-North [km]',
                                    ylabel='East-West [km]',
                                    toolbar='above',
                                    color='black',
                                    alpha=0.2,
                                    fill_color='black',
                                    fill_alpha=0.2,
                                    fontsize={'title': 20,
                                              'labels': 18,
                                              'xticks': 12,
                                              'yticks': 12}))

        # =====================================================================
        # Play/Pause botton.
        self.play_button = pn.widgets.Button(
            name='▶ Play', width=80, align='end')

        # =====================================================================
        # FUNCTION self.animate NEEDS UPDATE
        # self.play_button.on_click(self.animate)
        # =====================================================================

        # data is predefined variable in hv and it has to be supplied.
        # To do: find neat way to do this.
        self.stream_ring = hv.streams.Pipe(data=[])

        # =====================================================================
        # Evolution of hits based on time
        data_hits = np.array([np.array(self.hitX),
                              np.array(self.hitY),
                              np.array(self.peaktime),
                              np.array(self.peakamplitude),
                              self.palette_color])
        self.stream_hits = hv.streams.Pipe(data=data_hits)
        dmap_hits_plot = hv.DynamicMap(
            self.plot_hits, streams=[self.stream_hits]
        )
        # tap hits antenna to plot its electric field traces.
        dmap_hits_plot.opts(opts.Points(tools=['tap', 'hover']))

        # Shower core
        pcore = hv.DynamicMap(
            self.plot_core, streams=[self.stream_hits]
        )

        # Plot GP300 geometry and dynamic map of hits on the same canvas.
        self.dmap = antposplot*dmap_hits_plot*pcore

        # =====================================================================
        # Click on antennae with signal to view it's E-field trace
        stream_click = hv.streams.Selection1D(
            source=dmap_hits_plot, index=[int(self.nhits/2)])
        self.antEtrace = hv.DynamicMap(
            self.pick_trace, streams=[stream_click]).opts(
            'Curve', framewise=True, axiswise=True)
        self.antEtrace_h = hv.DynamicMap(self.pick_hilbert_trace, streams=[
                                         stream_click]).opts('Curve',
                                                             framewise=True,
                                                             axiswise=True)

        # =====================================================================
        # Cerenkov Ring
        self.cerenkov_grd = hv.DynamicMap(
            self.peak_amplitude_ground_plane,
            streams=[self.stream_ring]).opts('Image',
                                             framewise=True,
                                             axiswise=True)
        self.cerenkov_sp = hv.DynamicMap(
            self.peak_amplitude_shower_plane, streams=[
                self.stream_ring]).opts('Image',
                                        framewise=True,
                                        axiswise=True)
        self.cerenkov_ap = hv.DynamicMap(
            self.peak_amplitude_angular_plane, streams=[
                self.stream_ring]).opts('Image',
                                        framewise=True,
                                        axiswise=True)
        self.cerenkov_ang = hv.DynamicMap(
            self.peak_cerenkov_angle, streams=[
                self.stream_ring]).opts('Points',
                                        framewise=True,
                                        axiswise=True)

        # =====================================================================
        # Shower info
        self.shower_info = hv.DynamicMap(
            self.plot_text, streams=[self.stream_ring])

        # =====================================================================
        # Arrange final layout for display
        th = 100  # total height.
        tw = 150  # total width.
        dw = 70  # dmap width. main display plot.
        eh = 30  # height of electric field trace and hilbert envelop.
        ew = 58  # widht of electric field trace and hilbert envelop.
        lh = 12  # logo height.
        w2 = int((tw-dw)/4)  # 2d plot width.

        layout = pn.GridSpec(width=1500, height=main_height)
        layout[0:5, 0:7] = self.play_button  # "play" butoon
        layout[0:5, 7:50] = self.input_file  # "Browse" button
        layout[0:5, 51:70] = self.choose_color  # "choose color" button

        layout[6:th, 0:dw] = self.dmap  # Event display (footprint)
        layout[0:eh, dw:dw+ew] = self.antEtrace  # Electric field traces
        layout[eh:2*eh, dw:dw+ew] = self.antEtrace_h  # Hilbert envelop

        # Grand logo.
        layout[3:lh+5, dw+ew+8:tw-3] = 'logo_withoutbords.png'

        # Event Info.
        layout[lh+12:2*eh, dw+ew+3:tw] = self.shower_info
        layout[2*eh+3:th, dw:dw+w2+3] = self.cerenkov_sp
        layout[2*eh+3:th, dw+w2+3:dw+4+2*w2] = self.cerenkov_ap
        layout[2*eh+3:th, dw+4+2*w2:dw+4+3*w2] = self.cerenkov_grd
        layout[2*eh+3:th, dw+4+3*w2:tw] = self.cerenkov_ang

        pn.serve(layout, address='0.0.0.0', port=46813, show=False)
#        layout.show()


if __name__ == '__main__':

    hv.extension('bokeh', 'matplotlib')
    hv.plotting.mpl.MPLPlot.fig_latex = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gf",
        # Proposed geometry of GP300 detector. NEED UPDATE
        default="./GP300propsedLayout.dat",
        help="Geometry of antenna in GP300.")
    parser.add_argument(
        "--datadir",
        default="./../../scripts/data_test/"
        + "/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/",
        help="Provide path to the directory where data are stored.")
    args = parser.parse_args()

    geofile = args.gf
    datadir = args.datadir

    if datadir == "":
        raise Exception(
            "Provide path to your data directory."
            + "Run: python3 event_viewer_to_root.py --datadir <path>")

    # Size of plots
    main_width = 750  # widht of the main plot.
    main_height = 700  # height of the main plot
    side_width = 350  # width of trace plots.
    side_height = 300  # height of trace plots.
    img_width = 380  # width of side kXB, kX(kXB) image.
    img_height = 300  # height of side kXB, kX(kXB) image.

    color_options = ['Blues', 'Reds', 'RdBu_r', 'RdYlBu_r',
                     'RdYlGn_r', 'Wistia', 'YlGn', 'YlGnBu',
                     'autumn_r', 'cividis_r', 'coolwarm',
                     'copper_r', 'gist_earth_r', 'gnuplot_r',
                     'magma_r', 'mako_r', 'plasma_r', 'rainbow',
                     'seismic', 'summer_r', 'spring', 'terrain_r', 'turbo',
                     'viridis_r', 'vlag', 'winter_r', 'colorblind']

    eventviewer = EventViewer()
    eventviewer.view()
