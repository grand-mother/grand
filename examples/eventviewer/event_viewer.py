#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2021 GRAND Collaboration
contact: rkoirala@nju.edu.cn

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
SOFTWARE."""

class EventViewer:
	"""
	To Run:
		python3 event_viewer.py --datadir <path>
		python3 event_viewer.py --datadir <path> --hf <filename> --gf <geometry filename>

	Using "holoviews" software as a base for visualization and creating widget. Needs python 3.6 or higher.
	$ pip3 install "holoviews[all]"
		# Instruction: http://holoviews.org/
		put ~/.local/bin in the PATH so that packages installed there can be accessed.
		$ cd ~
		$ emacs -nw .bashrc
		Inside .bashrc, add the following line
			export PATH=/home/<your home directory>/.local/bin:$PATH
	
	holoview should install bokeh, if not install it using pip3
	$ pip3 install bokeh
		# Instruction: https://docs.bokeh.org/en/latest/
	"""
	def __init__(self):
		self.geofile = geofile
		self.hdffile = hdffile
		self.tstep   = 1500   # time step to look for antennae begin hit. If this time is short, event formation on viewer will take longer.
		self.fmin    = 50.e6  # minimum radio frequency in hertz.
		self.fmax    = 200.e6 # maximum radio frequency in hertz.
		self.plt_core = False

	def get_geometry(self):
		# Get layout of proposed geometry of GP300.
		self.geo_df  = pd.read_csv(self.geofile, sep=" ", usecols=[1,2,3,4])
		self.geo_df['ID'] = np.array([str(ant_name) for ant_name in self.geo_df['ID']]) # sometimes int are used as ID.
		self.posx    = self.geo_df['X']/1.e3     # x-coordinate of all antenna in km.
		self.posy    = self.geo_df['Y']/1.e3     # y-coordinate of all antenna in km.
		'''
		For 3D:
		self.posz    = self.geo_df['Z']/1.e3     # z-coordinate of all antenna in km.
		'''

	def get_data(self):
		# Collect information of an event required to make plots in event-display.
		# This method is called everytime a new data file is given as an input.
		self.run_info   = hdf5io.GetRunInfo(self.hdffile)
		event_num       = hdf5io.GetNumberOfEvents(self.run_info)-1
		self.eventname  = hdf5io.GetEventName(self.run_info,event_num)
		self.event_info = hdf5io.GetEventInfo(self.hdffile, self.eventname)
		self.ant_info   = hdf5io.GetAntennaInfo(self.hdffile, self.eventname)
		# filtered peak time and peak amplitude calculated after hilbert transform.
		self.peaktime, self.peakamplitude = mix.get_filtered_peakAmpTime_Hilbert(self.hdffile,
			self.eventname, 
			self.ant_info,
			self.fmin, self.fmax)		
		self.ant_info['peakamplitude']= self.peakamplitude
		sorted_indx     = self.ant_info.argsort('T0')          # index based on increasing time.
		sorted_antInfo  = self.ant_info[sorted_indx]
		self.hitAnt     = sorted_antInfo['ID']
		self.hitAnt     = np.array([str(ant_name) for ant_name in self.hitAnt])
		self.peakamplitude = sorted_antInfo['peakamplitude']
		# Match antennae name in geometry file with antennae name in event file.
		if ('A' in self.geo_df['ID'][0]) and ('A' not in self.hitAnt[0]):
			modified_ant_name = [ant_name[1:] for ant_name in self.geo_df['ID']]
			self.geo_df['ID'] = modified_ant_name
		elif ('A' not in self.geo_df['ID'][0]) and ('A' in self.hitAnt[0]):
			modified_ant_name = ['A'+ant_name for ant_name in self.geo_df['ID']]
			self.geo_df['ID'] = modified_ant_name
		'''
		Position of Antennae changes based on shower core. This gives variable antannae position if taken from AntennaeInfo. 
		So antennae position is taken from the geometry based on Antennae ID. These are position of hit antennae in GRAND coordinate system.
		'''		
		self.hitX       = np.array([(self.geo_df[self.geo_df.ID.eq(ant_name)]['X']).values[0] for ant_name in self.hitAnt])
		self.hitY       = np.array([(self.geo_df[self.geo_df.ID.eq(ant_name)]['Y']).values[0] for ant_name in self.hitAnt])
		self.hitZ       = np.array([(self.geo_df[self.geo_df.ID.eq(ant_name)]['Z']).values[0] for ant_name in self.hitAnt])
		# Position of hit antannae from shower core.
		self.hitXc      = sorted_antInfo['X']         #x-coordinate of hit antenna from shower core. Used to plot kXB, kX(kXB), K plot.
		self.hitYc      = sorted_antInfo['Y']         #y-coordinate of hit antenna from shower core. Used to plot kXB, kX(kXB), K plot.
		self.hitZc      = sorted_antInfo['Z']         #z-coordinate of hit antenna from shower core. Used to plot kXB, kX(kXB), K plot.		
		# Position of shower core in GRAND cs.
		self.corex     = (self.hitX-self.hitXc)[0]    #x-coordinate of shower core in GRAND cs. Not used.
		self.corey     = (self.hitY-self.hitYc)[0]    #y-coordinate of shower core in GRAND cs. Not used.
		#self.corez     = (self.hitZ-self.hitZc)[0]   #z-coordinate of shower core in GRAND cs. Not used.
		self.hitT       = sorted_antInfo['T0']
		self.peakA      = sorted_antInfo['peakamplitude']
		# More info related to the shower.
		self.primary    = hdf5io.GetEventPrimary(self.run_info, event_num)
		self.energy     = hdf5io.GetEventEnergy(self.run_info, event_num)	
		self.zenith     = np.deg2rad(hdf5io.GetEventZenith(self.run_info, event_num))
		self.azimuth    = np.deg2rad(hdf5io.GetEventAzimuth(self.run_info, event_num))
		self.bFieldIncl = np.pi/2. + np.deg2rad(hdf5io.GetEventBFieldIncl(self.event_info))
		self.bFieldDecl = -1.*np.deg2rad(hdf5io.GetEventBFieldDecl(self.event_info)) #0.
		self.x_xmax, self.y_xmax, self.z_xmax = hdf5io.GetXmaxPosition(self.event_info).data[0]
		self.slant_xmax = hdf5io.GetEventSlantXmax(self.run_info, event_num)
		# additional info.
		self.palette_color = self.select_color()
		self.tbins         = np.arange(min(self.hitT)-2*self.tstep, max(self.hitT)+2*self.tstep, self.tstep) # time boundary to look for hits.
		self.nhits         = len(self.hitX)

	def get_trace(self):
		lw = 1.5    # line-width of traces curves.
		alp = 0.9   # alpha on used color.

		tcurve = {}
		tcurve_h= {}
		peakA, peakT = np.zeros(self.nhits), np.zeros(self.nhits)
		for i, ant_id in enumerate(self.hitAnt):
			efield      = hdf5io.GetAntennaEfield(self.hdffile, self.eventname, ant_id,OutputFormat="numpy")
			efield[:,0]*= 1.e-9 #from ns to s. This is important for mix.filters(), if not error is produced.
			efield_filt = mix.filters(efield, FREQMIN=self.fmin, FREQMAX=self.fmax)
			hilbert_amp = np.abs(hilbert(efield_filt[1:4,:]))
			
			# plot traces
			curvex = hv.Curve(efield[:,1], 'Time Bins', 'E-field Trace', label='Ex')\
					.opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp, color='r')
			curvey = hv.Curve(efield[:,2], 'Time Bins', 'E-field Trace', label='Ey')\
					.opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp, color='steelblue')
			curvez = hv.Curve(efield[:,3], 'Time Bins', 'E-field Trace', label='Ez')\
					.opts(line_width=lw, tools=['hover'], xlabel='', alpha=alp, color='olive')
			curve  = curvex*curvey*curvez

			ymin = min([min(efield[:,1]), min(efield[:,2]), min(efield[:,3])])
			ymin = ymin - .05*abs(ymin)
			ymax = max([max(efield[:,1]), max(efield[:,2]), max(efield[:,3])])
			ymax = ymax + .05*abs(ymax)

			curve.opts(show_grid=True, title='Antenna: '+ant_id, 
					   toolbar='above',
					   xlim=(-1,len(efield[:,1])+1),
					   ylim=(ymin,ymax), # important to dynamically change ylim of traces. 
					   legend_position='top_right', 
					   legend_cols=3)
			tcurve[i] = curve

			# plot hilbert transform
			curvexh = hv.Curve(hilbert_amp[0,:], 'Time Bins', 'E-field [μV/m]')\
					.opts(line_width=lw, tools=['hover'], alpha=alp-0.1, color='r') #line_dash='dotted', 
			curveyh = hv.Curve(hilbert_amp[1,:], 'Time Bins', 'E-field [μV/m]')\
					.opts(line_width=lw, tools=['hover'], alpha=alp, color='steelblue')
			curvezh = hv.Curve(hilbert_amp[2,:], 'Time Bins', 'E-field [μV/m]')\
					.opts(line_width=lw, tools=['hover'], alpha=alp, color='olive')
			curve_h = curvexh*curveyh*curvezh

			ymin_h = min([min(hilbert_amp[0,:]), min(hilbert_amp[1,:]), min(hilbert_amp[2,:])])
			ymax_h = max([max(hilbert_amp[0,:]), max(hilbert_amp[1,:]), max(hilbert_amp[2,:])])
			ymin_h = ymin_h - .05*abs(ymax_h) # ymin is always 0.
			ymax_h = ymax_h + .05*abs(ymax_h)

			curve_h.opts(title='Hilbert Envelope',
						 show_grid=True, 
						 xlim=(-1,len(hilbert_amp[0,:])+1),
						 ylim=(ymin_h, ymax_h),
						 toolbar='above')#title='Hilbert Envelope',
			tcurve_h[i] = curve_h

		self.Eweight            = self.peakA + 15. # to prevent from taking log on <1 numbers. Use any number >=1.
		self.trace_collection   = tcurve
		self.hilbert_collection = tcurve_h

	def pick_trace(self, index):
		'''
		Electric field traces from all hit antennae are collected in 'trace_collection'. 
		This function picks trace from collected traces for a particular antenna for 
		plotting when clicked on that antenna. Traces are plotted only for antennae 
		that are hit.
		'''
		if not index:
			c1 = hv.Curve([], 'Time Bins', 'E-field Trace')
			c2 = hv.Curve([], 'Time Bins', 'E-field Trace')
			c3 = hv.Curve([], 'Time Bins', 'E-field Trace')
			curve = c1*c2*c3
			return curve

		antEtrace = self.trace_collection[index[0]]  #index here is a list with 1 entry.
		antEtrace.opts(width=side_width, height=side_height, show_grid=True,
			fontsize={'title':16, 'labels':13, 'legend':8, 'xticks':10, 'yticks': 10})

		return antEtrace

	def pick_hilbert_trace(self, index):
		'''
		Hilbert envelop of traces from all hit antennae are collected in 'hilbert_collection'. 
		This function picks trace from collected traces for a particular antenna for 
		plotting when clicked on that antenna. Traces are plotted only for antennae 
		that are hit.
		'''
		if not index:
			c1 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
			c2 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
			c3 = hv.Curve([], 'Time Bins', 'E-field [μV/m]')
			curve = c1*c2*c3
			return curve

		antEtrace_h = self.hilbert_collection[index[0]]  #index here is a list with 1 entry.
		antEtrace_h.opts(width=side_width, height=side_height, show_grid=True,
			fontsize={'title':10, 'labels':13, 'legend':8, 'xticks':10, 'yticks': 10})

		return antEtrace_h

	def plot_text(self,data=[]):
		# Print basic shower information on the display.
		# To Do: Extend this to include experimental events.
		quantity = ['Particle','Ene [EeV]','Zen [deg]','Azi [deg]','BInc [deg]', 'BDec [deg]', 'Xmax [g/cm2]']
		value    = [self.primary, 
	        		round(self.energy,2),
	        		round(np.rad2deg(self.zenith),2),
	        		round(np.rad2deg(self.azimuth),2),
	                round(np.rad2deg(self.bFieldIncl)-90.,2),
	                round(-1*np.rad2deg(self.bFieldDecl),2),
	                round(self.slant_xmax, 2)]

		text = {'Quantity': quantity, 'Value': value}
		df   = pd.DataFrame(text, columns = ['Quantity', 'Value'])
		txt_table = hv.Table(df)
		txt_table.opts(height=200, width=250)

		return txt_table

	def peak_amplitude_ground_plane(self, data):
		# plot interpolated peak amplitude in ground plane.
		X, Y = np.meshgrid(np.linspace(self.hitX.min(), self.hitY.max(), 200), 
						   np.linspace(self.hitY.min(), self.hitY.max(), 200))
		inter_peakamp_grd = scipolate.Rbf(self.hitX, self.hitY, self.peakamplitude, function='linear', epsilon=9)(X, Y) #function='thin_plate'
		kdims  = ['x_gp', 'y_gp']
		vdims  = ['peakA']
		bounds = (self.hitX.min()/1.e3, self.hitY.min()/1.e3, 
				  self.hitX.max()/1.e3, self.hitY.max()/1.e3)
		#np.flipud(data) is performed inside Image. So it is done here to undo that process. If not done, the image will be upside down.
		plot_cc = hv.Image(np.flipud(inter_peakamp_grd), kdims=kdims, vdims=vdims, bounds=bounds)  \
					.opts(width = img_width, 
						height  = img_height,
						cmap    = 'Spectral_r',
						title   = 'Ground Plane [km]',
						xlabel  = '',
						ylabel  = '',
						tools   = ['hover'],
						toolbar = 'below',
						fontsize= {'title': 10, 'labels': 11, 'xticks': 10, 'yticks': 10})
		return plot_cc

	def peak_amplitude_shower_plane(self, data):
		# Collect information of an event required to make plots in event-display.
		# This part of code came from Valentin Decoene.
		# 'data' is kept here only for syntax reason and is not used. Remove this in future.
		self.k_shower   = np.array([np.sin(self.zenith)*np.cos(self.azimuth), 
							   		np.sin(self.zenith)*np.sin(self.azimuth), 
							   		np.cos(self.zenith)])
		# Position of antannae in shower coordinate system. Note that hitx and hitX are different. hitX is time ordered hitx.
		self.x_sp, self.y_sp, self.z_sp = mix.get_in_shower_plane(np.array([self.hitXc, self.hitYc, self.hitZc]), 
													  			  self.k_shower, 
													  			  np.array([0,0,np.mean(self.hitZ)]), # z-value is not on the ground.
													  			  self.bFieldIncl, self.bFieldDecl)
		Xsp, Ysp = np.meshgrid(np.linspace(self.x_sp.min(), self.x_sp.max(), 200), 
							   np.linspace(self.y_sp.min(), self.y_sp.max(), 200))
		inter_peakamp = scipolate.Rbf(self.x_sp, self.y_sp, self.peakamplitude, 
									  function='thin_plate', epsilon=9)(Xsp, Ysp) #linear

		kdims  = ['x_sp', 'y_sp']
		vdims  = ['peakA']
		xmin   = self.x_sp.min()/1.e3
		xmax   = self.x_sp.max()/1.e3
		ymin   = self.y_sp.min()/1.e3
		ymax   = self.y_sp.max()/1.e3
		bounds = (xmin, ymin, xmax, ymax)
		lmax   = max([xmax-xmin, ymax-ymin])
		
		# np.flipud(data) is performed inside hv.Image. So it is done here to undo that process. 
		# If not done, the image will be upside down.
		plot_cc = hv.Image(np.flipud(inter_peakamp), kdims=kdims, vdims=vdims, bounds=bounds) \
					.opts(width = img_width, 
						height  = img_height,
						cmap    = 'Spectral_r',
						title   = 'Shower Plane',
						xlabel  = 'vxB [km]',
						ylabel  = 'vx(vxB) [km]',
						xlim    = (xmin-0.005*lmax,xmin+1.005*lmax), # this is for equal aspect ratio.
						ylim    = (ymin-0.005*lmax,ymin+1.005*lmax), # this is for equal aspect ratio.
						tools   = ['hover'],
						toolbar = 'below',
						fontsize= {'title': 10, 'labels': 11, 'xticks': 10, 'yticks': 10})
		return plot_cc

	def peak_amplitude_angular_plane(self, data):
		# Plot cerenkov ring in angular plane.
		# 'data' is kept here only for syntax reason and is not used. Remove this in future.
		# note: calculations done here is borrowed from Valentin Decone.
		XmaxA_x = self.hitXc- self.x_xmax
		XmaxA_y = self.hitYc - self.y_xmax
		XmaxA_z = self.hitZc - self.z_xmax
		
		obs   = np.array([XmaxA_x, XmaxA_y, XmaxA_z])
		l     = np.sqrt(XmaxA_x**2 + XmaxA_y**2 + XmaxA_z**2)
		u_ant = obs / l
		cosw  = np.dot(self.k_shower,u_ant)
		cosw[np.where(cosw>1.)] = 1. #sometimes value of cosine is 1.00000006 instead of 1.

		self.w = np.arccos(cosw)
		eta    = np.arctan2(self.y_sp,self.x_sp) #arctan2 chooses the quadrant properly.

		#self.x_angular = np.rad2deg(w)*np.sign(self.x_sp)
		#self.y_angular = np.rad2deg(w)*np.sign(self.y_sp)
		self.x_angular = np.rad2deg(self.w)*np.cos(eta)
		self.y_angular = np.rad2deg(self.w)*np.sin(eta)

		Xang, Yang = np.meshgrid(np.linspace(self.x_angular.min(), self.x_angular.max(), 200), 
							     np.linspace(self.y_angular.min(), self.y_angular.max(), 200)) #60
		inter_peakamp = scipolate.Rbf(self.x_angular, self.y_angular, self.peakamplitude, 
									  function='linear', epsilon=9)(Xang, Yang) #function='quintic'

		kdims  = ['x_ap', 'y_ap']
		vdims  = ['peakA']
		xmin    = self.x_angular.min()
		xmax    = self.x_angular.max()
		ymin    = self.y_angular.min()
		ymax    = self.y_angular.max()
		bounds  = (xmin, ymin, xmax, ymax)
		lmax = max([xmax-xmin, ymax-ymin])
		bounds  = (xmin, ymin, xmax, ymax)
		lmax = max([self.x_angular.max()-self.x_angular.min(), self.y_angular.max()-self.y_angular.min()])
		#np.flipud(data) is performed inside Image. So it is done here to undo that process. If not done, the image will be upside down.
		plot_cc = hv.Image(np.flipud(inter_peakamp), kdims=kdims, vdims=vdims, bounds=bounds)   \
					.opts(width = img_width, 
						height  = img_height,
						cmap    = 'Spectral_r',
						title   = 'Angular Plane',
						xlabel  = 'ω along vxB [deg]',
						ylabel  = 'ω along vx(vxB) [deg]',
						xlim    = (xmin-0.005*lmax,xmin+1.005*lmax), # this is for equal aspect ratio.
						ylim    = (ymin-0.005*lmax,ymin+1.005*lmax), # this is for equal aspect ratio.
						tools   = ['hover'],
						toolbar = 'below',
						fontsize= {'title': 10, 'labels': 11, 'xticks': 10, 'yticks': 10})
		return plot_cc

	def peak_cerenkov_angle(self,data):
		# Plot cerenkov ring in angular plane.
		# 'data' is kept here only for syntax reason and is not used. Remove this in future.
		kdims  = ['Omega', 'peakA']
		'''
		plot_ca1 = hv.Points(np.column_stack((self.x_angular, self.peakA)), kdims=kdims, label='kxB').opts(#width=img_width, height=img_height,
										 color='r', alpha=0.7,
										 tools=['hover'],
										 size=4
										 )
		plot_ca2 = hv.Points(np.column_stack((self.y_angular, self.peakA)), kdims=kdims, label='kx(kxB)').opts(#width=img_width, height=img_height,
										 color='g', alpha=0.7,
										 tools=['hover'],
										 size=4
										 )
		plot_ca = plot_ca1*plot_ca2
		'''
		x_omega = np.rad2deg(self.w)*np.sign(self.y_sp)
		plot_ca = hv.Points(np.column_stack((x_omega, self.peakA)), kdims=kdims) \
					.opts(#width=img_width, height=img_height,
						color = 'k', 
						alpha = 0.9,
						tools = ['hover'],
						size  = 4)

		#xmin = min([min(self.x_angular), min(self.y_angular)]) - 0.1
		#xmax = max([max(self.x_angular), max(self.y_angular)]) + 0.1
		xmin = min(x_omega) - 0.1
		xmax = max(x_omega) + 0.1
		ymin = min(self.peakA) * 1.05
		ymax = max(self.peakA) * 1.05
		plot_ca.opts(
			legend_position='bottom_right', 
			legend_cols = 3,
			toolbar     = 'below',
			show_grid   = True,
			xlabel      = 'ω [deg]',
			ylabel      = 'Peak Amp [μV/m]',
			xlim        = (xmin,xmax),
			ylim        = (ymin,ymax))
																						
		return plot_ca

	def animate(self, event):
		'''
		Controls Play button. Plot hits binned in time.
		'''
		if self.play_button.name == '▶ Play':
			self.play_button.name = '❚❚ Pause'
			# Check if the input file name has been changed. If changed start plotting the new event after 'Play' button is clicked.
			filename0  = self.input_file.filename	
			try:			
				if filename0!=None:
					findex0   = np.where('/'==np.array([i for i in filename0]))[0]
					if len(findex0)!=0:
						filename = datadir + filename0[findex0[-1]+1:]
					else:
						filename = datadir + filename0
					if filename!=self.hdffile:
						# if a new hdf file is provided, then start from the beginning.
						self.hdffile       = filename
						self.get_data()    # get hitX, hitY, ..., tbins etc for a new input hdf file.
						self.get_trace()   # get electric field traces from a new input hdf file.
						self.stream_ring.send(data=[]) # sending nothing, just calling to replot with updated data.
						
					if self.choose_color.value != self.select_color():
						self.get_data()
						self.stream_ring.send(data=[])
				self.plt_core = True
				indx      = 0
				# loop over all hits and send data via pipe to plot one by one.
				while indx<len(self.tbins):
					# It is faster to plot and hit-evolution looks smooth if hits are binned in time steps.
					mask = self.hitT<=self.tbins[indx]
					x    = np.array(self.hitX)[mask]           # select x-coordinate of hit antennae before a given time.
					y    = np.array(self.hitY)[mask]           # select y-coordinate of hit antennae before a given time.
					t    = np.array(self.hitT)[mask]           # select list of time before the boundary time.
					wt   = np.array(self.Eweight)[mask]        # Weight based on peak amplitude. This is an adhoc weight and has no physical meaning.
					color = np.array(self.palette_color)[mask] # select color from a palette that was created based on time of hit.
					self.stream_hits.send((x,y,t,wt,color))    # tunnel hits info to a dynamic map.
					indx+= 1
				# Show play button after an event is displayed.
				self.play_button.name = '▶ Play'
			except:
				# After all hits are plotted, change the 'Pause' button to 'Play'.
				print("ERROR: Choose a file to display event.")
				self.play_button.name = '▶ Play'

	def plot_hits(self,data):
		"""
		This function controls evolution of hits on detector geometry. Color represents the time of hit and 
		the size of circle represents the size of signal on antennae. But the size of circle and
		the strength of electric field/voltage on antennae are not directly related.
		
		Hits to be plotted are binned in time so that hits are evolved in a reasonable speed. 
		Note that the provided antennae information (x,y,t) are already sorted based by time of hit.

		'data' is sent here from 'animate' function inside a while loop.
		"""
		x = data[0]     # updated x-coordinate of hit tanks to be plotted.
		y = data[1]     # updated y-coordinate of hit tanks to be plotted.
		t = data[2]     # updated hit time of hit tanks to be plotted.
		wt= data[3]     # updated weight of hit tanks to be plotted.
		color = data[4] # updated colors to represent time of hit.
		kdims = ['X', 'Y']
		vdims = ['Weight', 'Time', 'Color']
		if len(x)==0: # if empty data is sent, this prevents code to fail and plots empty hits.
			fig = hv.Points([], kdims=kdims, vdims=vdims)
			fig.opts(opts.Points(width=main_width, height=main_height, tools=['hover']))
		else:
			pd_data = pd.DataFrame(data={'X':x/1.e3,  # m --> km.
										 'Y':y/1.e3,  # m --> km.
										 'Weight':wt, 
										 'Time':t,
										 'Color':color})

			ds = hv.Dataset(pd_data)                # create a holoview dataset from pandas dataframe.
			'''This is the part where hits are plotted. This function is called many times and number of 
			hits are added in each call until all hits are included.'''
			fig = hv.Points(ds, kdims=kdims, vdims=vdims)
			'''Add options to the plot.'''
			fig.opts(opts.Points(width=main_width, height=main_height,
								 size=np.log(dim('Weight'))*4., # signal strenght. This is arbitrary.
								 marker='circle',
								 color='Color',
								 alpha=0.95,
								 tools=['hover']))
		return fig

	def plot_core(self, data):
		if self.plt_core:
			return hv.Points((self.corex/1.e3, self.corey/1.e3)).opts(color='k', marker='star_dot', size=25)
		else:
			return hv.Points([]).opts(color='k', marker='star_dot', size=25)
			
			
	def select_color(self):
		self.color_pallete = sns.palettes.color_palette(self.choose_color.value, len(self.hitX)).as_hex()
		return self.color_pallete                	


	def view(self):
		# This is the driving function. All necessary process are called and managed from here.
		# Updating hits plot dynamically is done from here.
		
		# --------------Choose color ----------------
		self.choose_color = pn.widgets.Select(options=color_options, value='copper_r')

		# ----------- Browse event file to display -------------
		self.input_file = pn.widgets.FileInput(accept='.hdf5, .root')
		self.input_file.filename = self.hdffile
		self.get_geometry()# get updated position of antennae, (i.e. posx, posy)
		self.get_data()    # get updated hitAnt, hitX, hitY, hitT etc...
		self.get_trace()   # get updated electric field traces and hilbert envelop.		

		# -------- Plot detector geometry with all antennae position ---------
		antpos       = np.column_stack((self.posx, self.posy))
		antposplot   = hv.Points(antpos, kdims=['X','Y'])#.opts(fontsize={'xticks': 10, 'yticks': 10})
		antposplot.opts(opts.Points(width=main_width,
			height   = main_height,
			marker   = 'circle',
			size     = 8,
			tools    = ['hover'],
			xlabel   = 'South-North [km]', 
			ylabel   = 'East-West [km]',
			toolbar  = 'above',
			color    = 'black',
			alpha    = 0.2,
			fill_color = 'black',
			fill_alpha = 0.2,
			fontsize = {'title': 20, 'labels': 18, 'xticks': 12, 'yticks': 12}))

		# --------- Play/Pause botton. ------------------#
		self.play_button   = pn.widgets.Button(name='▶ Play', width=80, align='end')
		self.play_button.on_click(self.animate)
		self.stream_ring  = hv.streams.Pipe(data=[]) #data is predefined variable in hv and it has to be supplied. To do: find neat way to do this.
		# --------- Evolution of hits based on time ------------------
		self.stream_hits  = hv.streams.Pipe(data=[[],[],[],[],[]])
		dmap_hits_plot    = hv.DynamicMap(self.plot_hits, streams=[self.stream_hits])
		dmap_hits_plot.opts(opts.Points(tools=['tap', 'hover'])) # tap hits antenna to plot its electric field traces.
		pcore             = hv.DynamicMap(self.plot_core, streams=[self.stream_hits])
		self.dmap         = antposplot*dmap_hits_plot*pcore      # plot GP300 geometry and dynamic map of hits on the same canvas.
		
		# --------------Click on antennae with signal to view it's E-field trace ---------------------------
		stream_click      = hv.streams.Selection1D(source=dmap_hits_plot, index=[int(self.nhits/2)])
		self.antEtrace    = hv.DynamicMap(self.pick_trace, streams=[stream_click]).opts('Curve', framewise=True, axiswise=True)
		self.antEtrace_h  = hv.DynamicMap(self.pick_hilbert_trace, streams=[stream_click]).opts('Curve', framewise=True, axiswise=True)
		
		# ----- Cerenkov Ring -----------
		self.cerenkov_grd = hv.DynamicMap(self.peak_amplitude_ground_plane, streams=[self.stream_ring]).opts('Image', framewise=True, axiswise=True)
		self.cerenkov_sp  = hv.DynamicMap(self.peak_amplitude_shower_plane, streams=[self.stream_ring]).opts('Image', framewise=True, axiswise=True)
		self.cerenkov_ap  = hv.DynamicMap(self.peak_amplitude_angular_plane, streams=[self.stream_ring]).opts('Image', framewise=True, axiswise=True)
		self.cerenkov_ang = hv.DynamicMap(self.peak_cerenkov_angle, streams=[self.stream_ring]).opts('Points', framewise=True, axiswise=True)
		self.shower_info  = hv.DynamicMap(self.plot_text, streams=[self.stream_ring])

		# ----------- Arrange final layout for display -------------------
		th = 100 # total height.
		tw = 150 # total width.
		dh = 95  # dmap height. main display plot.
		dw = 70  # dmap width. main display plot.
		eh = 30  # height of electric field trace and hilbert envelop.
		ew = 58  # widht of electric field trace and hilbert envelop.
		lh = 12  # logo height.
		lw = 5  # logo width.
		bw = tw-(dw+ew)  # text box and logo width.
		bh = th-(2*eh + lh)  # text box height.
		h2 = th-(2*eh)       #2d plot height. not used.
		w2 = int((tw-dw)/3)  # 2d plot width.

		layout = pn.GridSpec(width=1500, height=main_height)#, sizing_mode='stretch_both')
		layout[0:5, 0:7]  = self.play_button        # "play" butoon
		layout[0:5, 7:50] = self.input_file         # "Browse" button
		layout[0:5,51:70] = self.choose_color       # "choose color" button
		layout[6:th,0:dw] = self.dmap               # main event display
		layout[0:eh, dw:dw+ew] = self.antEtrace        # Electric field traces
		layout[eh:2*eh, dw:dw+ew] = self.antEtrace_h   # Hilbert envelop
		layout[3:lh+5, dw+ew+8:tw-3] = 'logo_withoutbords.png'        # Grand logo.
		layout[lh+12:2*eh, dw+ew+3:tw] = self.shower_info   # Event Info.
		layout[2*eh+3:th, dw:dw+w2+3] = self.cerenkov_sp
		layout[2*eh+3:th, dw+w2+3:dw+4+2*w2] = self.cerenkov_ap
		layout[2*eh+3:th, dw+4+2*w2:tw] = self.cerenkov_ang
		#layout[2*eh+3:th, dw:dw+w2+3] = self.cerenkov_grd
		#layout[2*eh+3:th, dw+w2+3:dw+4+2*w2] = self.cerenkov_sp
		#layout[2*eh+3:th, dw+4+2*w2:tw] = self.cerenkov_ap

		layout.show()

if __name__=='__main__':

	import argparse
	import re, os
	import glob
	import numpy as np
	import pandas as pd    
	# http://holoviews.org/getting_started/index.html
	import panel as pn
	import holoviews as hv
	from holoviews import opts, dim
	import hdf5fileinout as hdf5io      # written by Matias Tueros.
	
	from scipy.signal import hilbert
	import scipy.interpolate as scipolate
	import mix                          # functions wrtten by Valentin Decoene.
	import seaborn as sns               # used for color pallettes.

	hv.extension('bokeh', 'matplotlib')
	hv.plotting.mpl.MPLPlot.fig_latex=True

	# =============================================================================
	parser = argparse.ArgumentParser()
	parser.add_argument("--gf",
			  # Proposed geometry of GP300 detector. 
			  # from => @cca.in2p3.fr:/sps/hep/trend/tueros/DiscreteGP300LibraryDunhuang/GP300propsedLayout.dat
			  default="Data/GP300propsedLayout.dat", 
			  help="Geometry of antenna in GP300.")
	parser.add_argument("--hf",
			  default="Data/GP300_Proton_0.1_63.0_20.77_10.hdf5",
			  help="Name of one or many hdf5 file where experimental or simulation data are stored.")
	parser.add_argument("--datadir",
			  default="", 
			  help="Provide path to the directory where data are stored.")

	args       = parser.parse_args()

	geofile    = args.gf
	hdffile    = args.hf
	datadir    = args.datadir

	if datadir=="":
		raise Exception("Provide path to your data directory. Run: python3 event_viewer.py --datadir <path>")

	# Size of plots
	main_width  = 750 # widht of the main plot.
	main_height = 700 # height of the main plot
	side_width  = 350 # width of trace plots.
	side_height = 300 # height of trace plots.
	img_width   = 380 # width of side kXB, kX(kXB) image.
	img_height  = 300 # height of side kXB, kX(kXB) image.

	color_options = ['Blues', 'Reds', 'RdBu_r', 'RdYlBu_r', 
					 'RdYlGn_r', 'Wistia', 'YlGn', 'YlGnBu', 
					 'autumn_r', 'cividis_r', 'coolwarm', 
					 'copper_r', 'gist_earth_r', 'gnuplot_r', 
					 'magma_r', 'mako_r', 'plasma_r', 'rainbow',
					 'seismic', 'summer_r', 'spring', 'terrain_r', 'turbo', 
					 'viridis_r', 'vlag', 'winter_r', 'colorblind']

	# all options
	'''color_options = ['Accent', 'Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn',
					 'BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r','Dark2','Dark2_r',
					 'GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','OrRd',
					 'OrRd_r','Oranges','Oranges_r','PRGn','PRGn_r','Paired','Paired_r',
					 'Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu',
					 'PuBuGn','PuBuGn_r','PuBu_r','PuOr','PuOr_r','PuRd','PuRd_r','Purples',
					 'Purples_r','RdBu','RdBu_r','RdGy','RdGy_r','RdPu','RdPu_r',
					 'RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Reds','Reds_r','Set1','Set1_r',
					 'Set2','Set2_r','Set3','Set3_r','Spectral','Spectral_r','Wistia',
					 'Wistia_r','YlGn','YlGnBu','YlGnBu_r','YlGn_r','YlOrBr','YlOrBr_r',
					 'YlOrRd','YlOrRd_r','afmhot','afmhot_r','autumn','autumn_r','binary',
					 'binary_r','bone','bone_r','brg','brg_r','bwr','bwr_r','cividis','cividis_r',
					 'cool','cool_r','coolwarm','coolwarm_r','copper','copper_r','cubehelix',
					 'cubehelix_r','flag','flag_r','gist_earth','gist_earth_r','gist_gray',
					 'gist_gray_r','gist_heat','gist_heat_r','gist_ncar','gist_ncar_r',
					 'gist_rainbow','gist_rainbow_r','gist_stern','gist_stern_r','gist_yarg',
					 'gist_yarg_r','gnuplot','gnuplot2','gnuplot2_r','gnuplot_r','gray','gray_r',
					 'hot','hot_r','hsv','hsv_r','icefire','icefire_r','inferno','inferno_r',
					 'magma','magma_r','mako','mako_r','nipy_spectral','nipy_spectral_r',
					 'ocean','ocean_r','pink','pink_r','plasma','plasma_r','prism','prism_r',
					 'rainbow','rainbow_r','rocket','rocket_r','seismic','seismic_r','spring',
					 'spring_r','summer','summer_r','tab10','tab10_r','tab20','tab20_r','tab20b',
					 'tab20b_r','tab20c','tab20c_r','terrain','terrain_r','turbo','turbo_r',
					 'twilight','twilight_r','twilight_shifted','twilight_shifted_r','viridis',
					 'viridis_r','vlag','vlag_r','winter','winter_r']

	# To see how these color pallettes look, Run this code (ipython)
	import seaborn as sns
	import matplotlib.pyplot as plt
	for item in color_options:
		print(item)
        current_palette = sns.color_palette(item, 100)
        sns.palplot(current_palette)
        plt.title(item)
        plt.show()
	'''
	eventviewer = EventViewer()
	eventviewer.view()


#
