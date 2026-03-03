from _importer import *

wing = trapezoidal_wing(surface_area=1, aspect_ratio=7, taper_ratio=1, twist=0, sweep=0, airfoil=NACA4("2400"))
wing.calculate_wing_parameters()
functions = lambda x: (3*x)*(abs(x)<0.3) + np.sign(x)*(0.9+(abs(x)-0.3)/7)*(abs(x)>=0.3)
wing.mesh(Nb=21, Nc=20, mesh_type_span="cosine", mesh_type_chord="uniform", simetric=True)
wing.plot_nodes()