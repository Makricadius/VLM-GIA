from geometry import *
from singularities import *
from aerodynamics import *
from post_processing import *

geometric_wing = trapezoidal_simetrical_wing(superficie=150, alargamiento=3, estrechamiento=0.1, torsión=-7, flecha=40, diedro=10)
ALA = Aerdynamic_wing(geometric_wing, Vortex_shoe)
ALA.generate_model(32)

density = 10
[ALA.calculate(alpha=e/density) for e in range(-5*density,5*density+1)]

geometric_wing.plot_nodes()
plot_aero_characteristics(ALA, show=True)

plot_streamlines_3d(ALA, alpha=7.0, stream_box_scale=(1.5, 1.2, 0.5), smoke_scale=(1.1, 0.2), stream_density=(11, 5), grid_resolution=(30, 23, 10), upstream_offset=0, max_length=1, step_size=0.04, show=True)