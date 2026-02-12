from math_functions import *

class Aerdynamic_wing:
    def __init__(self, wing_geometry, singularity_model):
        self.wing = wing_geometry
        self.singularity = singularity_model
    
    def generate_model(self):
        self.alpha_memory = dict()
        self.wing.calculate_wing_parameters()
        self.wing.mesh()
        self.model = self.singularity(self.wing.panels)
        self.model.geometry()
        self.model.matrix_A()
    
    def solve(self, alpha):
        B = rad(alpha)+self.model.ic
        self.model.circulations = - B @ self.model.inv_A.T
    
    def calculate(self, alpha):
        if alpha not in [e for e in self.alpha_memory]:
            self.wing.solve(alpha)
            self.alpha_memory[alpha] = dict()
            self.alpha_memory[alpha]["C"] = self.model.panels_data.circulations
            self.alpha_memory[alpha]["A"] = self.model.aerodinamic_characteristic()