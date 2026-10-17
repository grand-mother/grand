import ast
import inspect

from matplotlib.mathtext import MathTextParser
from grand.sim.noise import Compute_Plot_Galactic_Noise

# The y label of the galactic_bnu plot, as written in this version's source
line = next(l for l in inspect.getsource(Compute_Plot_Galactic_Noise.main).splitlines() if "<B$_{v}$>" in l)
label = ast.literal_eval(line.split("plt.ylabel(", 1)[1].rsplit(", fontsize", 1)[0])
print("label:", label)
try:
    MathTextParser("path").parse(label)   # what plt.savefig does with it
    print("matplotlib can draw it")
except ValueError as err:
    print("ValueError:", str(err).strip().splitlines()[-1])
