from grand.aoi.antenna import Antenna
from grand.aoi.event import Event
from grand.aoi.shower import Shower
from grand.aoi.timetrace import Voltage
from grand.sim.shower.gen_shower import ShowerEvent


def assign(make, field, value):
    """Sets the field on a new object; prints what was stored or the error."""
    label = "%s().%s = %r" % (make.__name__, field, value)
    try:
        obj = make()
        setattr(obj, field, value)
        print(label, "-> stored", repr(getattr(obj, field)))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


assign(Antenna, "id", -5)
assign(Event, "run_number", -1)
assign(Shower, "primary_type", 2212)
assign(Shower, "zenith", 200.0)
assign(Voltage, "n_points", -1)
assign(Voltage, "t_bin_size", -0.5)
assign(ShowerEvent, "primary", 2212)
assign(ShowerEvent, "primary", "Proton")
assign(ShowerEvent, "zenith", 200.0)
