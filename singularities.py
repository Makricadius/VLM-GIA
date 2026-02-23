import numpy as np

class Vortex_shoe:
    def __init__(self, panels, Nb, Nc):
        self.panels = panels
        self.Nb = Nb
        self.Nc = Nc

    def geometry(self):
        self.N = len(self.panels.xA)
        self.x1 = (3*self.panels.xA+self.panels.xC)/4
        self.y1 = (3*self.panels.yA+self.panels.yC)/4
        self.z1 = (3*self.panels.zA+self.panels.zC)/4
        self.x2 = (3*self.panels.xB+self.panels.xD)/4
        self.y2 = (3*self.panels.yB+self.panels.yD)/4
        self.z2 = (3*self.panels.zB+self.panels.zD)/4
        self.xc = (self.panels.xA+self.panels.xB+3*self.panels.xC+3*self.panels.xD)/8
        self.yc = (self.panels.yA+self.panels.yB+3*self.panels.yC+3*self.panels.yD)/8
        self.zc = (self.panels.zA+self.panels.zB+3*self.panels.zC+3*self.panels.zD)/8
        ABx2 = (self.panels.xA+self.panels.xB)/2
        CDx2 = (self.panels.xC+self.panels.xD)/2
        ABz2 = (self.panels.zA+self.panels.zB)/2
        CDz2 = (self.panels.zC+self.panels.zD)/2
        self.ic = -np.arctan2(CDz2-ABz2, CDx2-ABx2) # inclination
        self.cij = abs(self.panels.xD-self.panels.xB+self.panels.xC-self.panels.xA)/2
        self.bij = abs(self.panels.yA-self.panels.yB+self.panels.yC-self.panels.yD)/2
        self.sij = self.cij*self.bij
        self.x14ij = (self.panels.xA*3+self.panels.xB*3+self.panels.xC+self.panels.xD)/8
        self.circulations = None

        r2A = (np.array([self.y1]).T-self.yc)**2 + (-np.array([self.z1]).T+self.zc)**2
        self.w_libres_up   =  1/(2*np.pi*r2A)*( np.array([self.y1]).T-self.yc)
        self.v_libres_up   =  1/(2*np.pi*r2A)*(-np.array([self.z1]).T+self.zc)
        r2B = (np.array([self.y2]).T-self.yc)**2 + (-np.array([self.z2]).T+self.zc)**2
        self.w_libres_down =  1/(2*np.pi*r2B)*( np.array([self.y2]).T-self.yc)
        self.v_libres_down =  1/(2*np.pi*r2B)*(-np.array([self.z2]).T+self.zc)

    def get_w_inducido(self):
        self.w_inducido = np.sum(np.reshape(self.circulations @ self.w_libres_up - self.circulations @ self.w_libres_down, (self.Nc,self.Nb)), axis=0)
        self.v_inducido = np.sum(np.reshape(self.circulations @ self.v_libres_up - self.circulations @ self.v_libres_down, (self.Nc,self.Nb)), axis=0)
    
    def aerodinamic_characteristic(self, S, cam, xca):
        clij = np.reshape(2*self.circulations/self.cij,(self.Nc,self.Nb))
        sij = np.reshape(self.sij,(self.Nc,self.Nb))
        cij = np.reshape(self.cij,(self.Nc,self.Nb))
        clj = np.sum(clij*sij,axis=0)/np.sum(sij,axis=0)
        cl = np.sum(clij*sij)/S

        cm0y_ij = -clij*np.reshape(self.x14ij/self.cij,(self.Nc,self.Nb))
        scij = np.reshape(self.cij*self.sij,(self.Nc,self.Nb))
        cm0y_j = np.sum(cm0y_ij*scij,axis=0)/(np.sum(cij,axis=0)*np.sum(sij,axis=0))
        cm0y = np.sum(cm0y_ij*scij)/(S*cam)

        cma = cm0y + cl*xca/cam
        cmc4 = cm0y + cl*0.25
        if cl != 0:
            xcp = -cam*cm0y/cl
        else:
            xcp = np.inf

        self.get_w_inducido()
        w_inducido = self.w_inducido
        cdj = -clj*w_inducido/2
        cd = np.sum(cdj*sij)/S
        return {"cl":[cl,clj,clij],"cm":[cm0y,cm0y_j,cm0y_ij,cma,cmc4],"xcp":xcp,"cd":[cd,cdj,w_inducido]}
    
    def matrix_A(self):
        a = np.array([self.xc]).T-self.x1
        b = np.array([self.yc]).T-self.y1
        c = np.array([self.xc]).T-self.x2
        d = np.array([self.yc]).T-self.y2
        e = np.sqrt(a**2+b**2)
        f = np.sqrt(c**2+d**2)
        g = self.x2-self.x1
        h = self.y2-self.y1
        k = (g*a+h*b)/e-(g*c+h*d)/f
        l = -(1+a/e)/b+(1+c/f)/d
        self.A = k/(4*np.pi*(a*d-c*b))+l/(4*np.pi)
        self.condicionamiento = np.min(abs(self.A.diagonal())/np.max(abs(self.A),axis=1))
        self.inv_A = np.linalg.inv(self.A)
