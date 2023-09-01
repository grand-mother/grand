#!/usr/bin/python
# This is an example for displaying traces using the Analysis Oriented Interface. Works only for the first GP13 data.

from grand.grandlib_classes.grandlib_classes import *
import sys
import ROOT

def main():

	# Read the file name from command line
	if len(sys.argv)>1: file_name = sys.argv[1]
	# Or use a default GP13 file
	else: file_name = "GRAND.TEST-RAW.20230307174423.001.root"

	print("Reading file", file_name)

	# Create the EventList with the specified file, and tell it to use TRawVoltage tree as the voltage source
	el = EventList(file_name, use_trawvoltage=True)

	# First just get the first event, to draw the layout
	e = el.get_event()
	# Generating convenient array from antenna positions
	positions = np.array([[ant.position.x[0], ant.position.y[0]] for ant in e.antennas])

	# ROOT drawing of positions
	c2 = arrays2canvas(positions[:, 0], positions[:, 1])
	ROOT.gPad.Modified()
	ROOT.gPad.Update()

	# ROOT canvas for traces
	c1 = ROOT.TCanvas("c", "c", 1500, 500)

	# Iterate through all the events
	for i,e in enumerate(el):
		print(f"Event {i}, du_id {e.voltages[0].du_id}, time {e.voltages[0].t0}")

		# Get the traces
		t = e.voltages[0].t_vector
		x = e.voltages[0].trace.x
		y = e.voltages[0].trace.y
		z = e.voltages[0].trace.z
		
		# Turn traces into plots
		gx = arrays2graph(t, x)
		gy = arrays2graph(t, y)
		gz = arrays2graph(t, z)
		
		gz.SetTitle(f"Event {i}, du_id {e.voltages[0].du_id}, time {e.voltages[0].t0}")
		gz.GetXaxis().SetTitle("Time [ns]")
		gz.GetYaxis().SetTitle("Raw voltage [V]")
		
		gz.Draw("AL")
		gx.SetLineColor(2)
		gx.Draw("same L")
		gy.SetLineColor(3)
		gy.Draw("same L")		
		ROOT.gPad.Modified()
		ROOT.gPad.Update()

		# Wait for the user to press a key
		wait4key()

	
## Creates a TGraph from 2 1D numpy arrays and then draws it in a Canvas
def arrays2canvas(x, y=None, dx=None, dy=None, drawoption = "A*", z=None, dz=None, x_shift=0):
	g = arrays2graph(x, y, dx, dy, z, dz, x_shift)
	c = ROOT.TCanvas()
	g.Draw(drawoption)
	c.graph = g

	return c
	
## Creates a TGraph from 2 1D numpy arrays. Assuming the same length of the arrays.
def arrays2graph(x, y=None, dx=None, dy=None, z=None, dz=None, x_shift=0):
	# Normal case
	if y is not None:
		x = np.array(x)
		y = np.array(y)
	# If no y given, x is y, and x should be ordinal points for all y
	else:
		y = np.arange(len(x))
		x = np.array(x)
		x,y = y,x

	x+=x_shift

	g = None

	if z is None:
		if dx is not None: dx = np.array(dx)
		if dy is not None: dy = np.array(dy)

		if dx is None and dy is None:
			g = ROOT.TGraph(len(x), x.astype(np.float64), y.astype(np.float64))
		elif dx is None and not dy is None:
			dx = np.zeros(len(x))
			g = ROOT.TGraphErrors(len(x), x.astype(np.float64), y.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64))
		elif dy is None and not dx is None:
			dy = np.zeros(len(x))
			g = ROOT.TGraphErrors(len(x), x.astype(np.float64), y.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64))

	else:
		z = np.array(z)
		if dz is not None:
			dz = np.array(dz).astype(np.float64)
		if dx is None and dy is None and dz is None:
			g = ROOT.TGraph2D(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64))
		elif dx is None and not dy is None:
			dx = np.ones(len(x))
			g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64), dz)
		elif dy is None and not dx is None:
			dy = np.ones(len(x))
			g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64), dz)
		elif not dy is None and not dx is None:
			g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), dx.astype(np.float64), dy.astype(np.float64), dz)
		else:
			dx = np.ones(len(x))
			dy = np.ones(len(x))
			g = ROOT.TGraph2DErrors(len(x), x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), dx, dy, dz)

	g.SetTitle()
	return g
	
def wait4key():
	rep = ''
	br = False
	while not rep in [ 'q', 'Q', "x", "X", "c", "C" ]:
		# ROOT.gSystem.ProcessEvents()
		rep = input('enter "c" or "q" to continue, "x" to quit, "b" to break the loop: ')
		if len(rep)<1: continue
		if rep[0] in ['x','X']:
			sys.exit()
		if rep[0] in ['b','B']:
			br=True
			break
		if 1 < len(rep):
			rep = rep[0]
	if br: return True
	else: return False
	
	
if __name__ == '__main__':
	a = main()
	
	
