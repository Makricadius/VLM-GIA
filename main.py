from geometry import *
from singularities import *
from aerodynamics import *

geometric_wing = trapezoidal_simetrical_wing(superficie=100, alargamiento=6, estrechamiento=0, torsión=-3, flecha=35, diedro=0)
ALA = Aerdynamic_wing(geometric_wing, Vortex_shoe)
ALA.generate_model()
ALA.solve(alpha=5)
print(ALA.alpha_memory[5]["A"])
