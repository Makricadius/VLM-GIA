from src import *

geometric_wing = trapezoidal_simetrical_wing(superficie=100, alargamiento=4, estrechamiento=0.1,
                                             torsión=-5, flecha=40, diedro=0, 
                                             airfoil=NACA4("6420"))

# geometric_wing = trapezoidal_simetrical_wing(superficie=100, alargamiento=15, estrechamiento=0.5, torsión=-5, flecha=15, diedro=0)
# geometric_wing = trapezoidal_simetrical_wing(superficie=1000, alargamiento=1000, estrechamiento=1, torsión=0, flecha=0, diedro=0, airfoil=NACA4("5245"))

ALA = Aerdynamic_wing(geometric_wing, Vortex_shoe)
ALA.generate_model(200, 10)

ALA.wing.print_parameters()
geometric_wing.plot_nodes()


density = 17
[ALA.calculate(alpha=e/density) for e in range(-8*density,8*density+1)]
plot_aero_characteristics(ALA, show=True)
#plot_streamlines_3d(ALA, alpha=7.0, stream_box_scale=(1.5, 1.2, 0.5), smoke_scale=(1.1, 0.2), stream_density=(11, 5), grid_resolution=(30, 23, 10), upstream_offset=0, max_length=1, step_size=0.04, show=True)