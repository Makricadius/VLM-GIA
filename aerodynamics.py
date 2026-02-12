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
    
    
    def aerodinamic_characteristic(self):
        clij = np.reshape(2*self.model.circulations/self.model.cij,(self.model.Np_c,self.model.Nb))
        s_ij = np.reshape(self.model.sij,(self.model.Np_c,self.model.Nb))
        clj = np.sum(clij*s_ij,axis=0)/np.sum(s_ij,axis=0)
        cl = np.sum(clij*s_ij)/self.wing.S

        cm0y_ij = -clij*np.reshape(self.model.x14ij/self.model.cij,(self.model.Np_c,self.model.Nb))
        scij = np.reshape(self.model.cij*self.model.sij,(self.model.Np_c,self.model.Nb))
        scj = np.reshape(self.model.sij**2/self.model.bij,(self.model.Np_c,self.model.Nb))
        cm0y_j = np.sum(cm0y_ij*scij,axis=0)/np.sum(scj,axis=0)
        cm0y = np.sum(cm0y_ij*scij)/(self.wing.S*self.wing.cam)

        cma = cm0y + cl*self.wing.xca/self.wing.cam
        cmc4 = cm0y + cl*0.25
        xcp = -self.wing.cam*cm0y/cl

        self.model.get_w_inducido()
        w_inducido = self.model.w_inducido
        cdj = -clj*w_inducido/2
        cd = np.sum(cdj*s_ij)/self.wing.S
        return {"cl":[cl,clj,clij],"cm":[cm0y,cm0y_j,cm0y_ij,cma,cmc4],"xcp":xcp,"cd":[cd,cdj,w_inducido]}

    def solve(self, alpha):
        B = rad(alpha)+self.model.ic
        self.model.circulations = - B @ self.model.inv_A.T
    
    def calculate(self, alpha):
        if alpha not in [e for e in self.alpha_memory]:
            self.solve(alpha)
            self.alpha_memory[alpha] = dict()
            self.alpha_memory[alpha]["C"] = self.model.circulations
            self.alpha_memory[alpha]["A"] = self.aerodinamic_characteristic()