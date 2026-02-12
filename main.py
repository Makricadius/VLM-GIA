from geometry import *
from singularities import *
from aerodynamics import *
from post_processing import plot_aero_characteristics

geometric_wing = trapezoidal_simetrical_wing(superficie=200, alargamiento=11, estrechamiento=0.5, torsión=-3, flecha=15, diedro=0)
ALA = Aerdynamic_wing(geometric_wing, Vortex_shoe)
ALA.generate_model()

density = 10
[ALA.calculate(alpha=e/density) for e in range(-5*density,5*density)]

geometric_wing.plot_nodes()
plot_aero_characteristics(ALA, show=True)