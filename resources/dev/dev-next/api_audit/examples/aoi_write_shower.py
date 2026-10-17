import numpy as np
from grand import CartesianRepresentation
from grand.aoi import Event, Shower
from grand.dataio import TShower


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def point(x, y, z):
    return CartesianRepresentation(x=np.array([x]), y=np.array([y]), z=np.array([z]))


event = Event()
event.run_number, event.event_number = 1, 1
event.shower = Shower()
event.shower.energy_em = event.shower.energy_primary = 1.0
event.shower.Xmax = 700.0
event.shower.zenith, event.shower.azimuth = 10.0, 10.0
event.shower.Xmaxpos = point(1000.0, 0.0, 5000.0)       # Xmax seen from the shower core
event.shower.core_ground_pos = point(100.0, 0.0, 0.0)   # the core in the site frame
event.write_shower("shower.root")

tshower = TShower("shower.root")
tshower.get_entry(0)
print("in shower.root: xmax_pos", [float(v) for v in tshower.xmax_pos],
      "| xmax_pos_shc", [float(v) for v in tshower.xmax_pos_shc], "| analysis_level", tshower.analysis_level)

# The same Event object filling the shower tree for two events in turn
attempt("fill_shower_tree() for event 1", lambda: event.fill_shower_tree() or "filled")
event.event_number = 2
attempt("fill_shower_tree() for event 2", lambda: event.fill_shower_tree() or "filled")
