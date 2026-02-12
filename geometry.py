from math_functions import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Wing:
    def __init__(self, type):
        self.type = type

class NACA4:
    def __init__(self, naca_code):
        self.naca_code = naca_code
        self.f = int(self.naca_code[0])/100
        self.xf = int(self.naca_code[1])/10
        self.t = int(self.naca_code[2:])/100
    
    def camber_line(self, x):
        f1 = self.f/self.xf**2*(2*self.xf*x-x**2)
        f2 = self.f/(1-self.xf)**2*((1-2*self.xf)+2*self.xf*x-x**2)
        return f1*(x<self.xf)+f2*(x>self.xf)
    
    def thickness_distribution(self, x):
        thickness = 5*self.t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        return thickness*(1-x)**0.01

class Panels:
    def __init__(self, xA, yA, zA, xB, yB, zB, xC, yC, zC, xD, yD, zD):
        self.xA = xA
        self.yA = yA
        self.zA = zA
        self.xB = xB
        self.yB = yB
        self.zB = zB
        self.xC = xC
        self.yC = yC
        self.zC = zC
        self.xD = xD
        self.yD = yD
        self.zD = zD

class trapezoidal_simetrical_wing:
    def __init__(self, superficie=1, alargamiento=1, estrechamiento=1, torsión=0, flecha=0, diedro=0):
        self.S = superficie
        self.A = alargamiento
        self.estrechamiento = estrechamiento
        self.torsion = rad(torsión) # tip torsión
        self.flecha = rad(flecha) # sweep at c/4
        self.diedro = rad(diedro) # dihedral angle
    
    def calculate_wing_parameters(self):
        self.b = np.sqrt(self.A*self.S)
        self.cgm = self.b/self.A
        self.cr = 2*self.cgm/(1+self.estrechamiento)
        self.ct = self.cr*self.estrechamiento
        self.cam = 2/3*self.cr*(1+self.estrechamiento+self.estrechamiento**2)/(1+self.estrechamiento)
        self.yca = self.b/6*(1+2*self.estrechamiento)/(1+self.estrechamiento)
        self.xca = 0.25*self.cr+np.tan(self.flecha)*self.yca
    
    def mesh(self, Nb=20):
        """Defining the nodes span-wise"""
        y_theta = np.linspace(0, np.pi, Nb+1)
        y_nodes = -np.cos(y_theta)*self.b/2
        y_scaling = 2*abs(y_nodes)/self.b
        """Defining the nodes coord-wise"""
        x_coord_nodes = np.array([[0], [1]])
        x_coord = self.cr + (self.ct-self.cr)*y_scaling
        """Generating X & Y meshes for the nodes"""
        mesh = np.ones((2,Nb+1))
        x_mesh = (x_coord_nodes*mesh-0.25)*x_coord
        y_mesh = mesh*y_nodes
        z_mesh = np.zeros((2,Nb+1))
        """"Applying the twist to the nodes"""
        self.linear_torsion = self.torsion*y_scaling
        x_mesh, z_mesh = rotate(x_mesh, z_mesh, -self.linear_torsion)
        """Applying the sweepto the nodes"""
        x_mesh += np.tan(self.flecha)*abs(y_nodes)
        """"Applying the dihedral angle to the nodes"""
        y_mesh, z_mesh = rotate(y_mesh, z_mesh, self.diedro*np.sign(y_nodes))
        """Leading edge of root coord at 0,0,0,"""
        x_mesh += 0.25*self.cr
        """Storing the nodes in the class"""
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh
        
        xA = np.reshape(self.x_mesh[:-1,:-1], Nb)
        xB = np.reshape(self.x_mesh[:-1,1:], Nb)
        xC = np.reshape(self.x_mesh[1:,:-1], Nb)
        xD = np.reshape(self.x_mesh[1:,1:], Nb)
        yA = np.reshape(self.y_mesh[:-1,:-1], Nb)
        yB = np.reshape(self.y_mesh[:-1,1:], Nb)
        yC = np.reshape(self.y_mesh[1:,:-1], Nb)
        yD = np.reshape(self.y_mesh[1:,1:], Nb)
        zA = np.reshape(self.z_mesh[:-1,:-1], Nb)
        zB = np.reshape(self.z_mesh[:-1,1:], Nb)
        zC = np.reshape(self.z_mesh[1:,:-1], Nb)
        zD = np.reshape(self.z_mesh[1:,1:], Nb)

        self.panels = Panels(xA, yA, zA, xB, yB, zB, xC, yC, zC, xD, yD, zD)
    
    def plot_nodes(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(self.x_mesh.shape[0]):
            ax.plot(self.x_mesh[i, :], self.y_mesh[i, :], self.z_mesh[i, :], 'b-', alpha=0.6)
        
        for j in range(self.x_mesh.shape[1]):
            ax.plot(self.x_mesh[:, j], self.y_mesh[:, j], self.z_mesh[:, j], 'b-', alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Wing Nodes')
        
        max_range = np.array([self.x_mesh.max()-self.x_mesh.min(), self.y_mesh.max()-self.y_mesh.min(), self.z_mesh.max()-self.z_mesh.min()]).max() / 2.0
        mid_x = (self.x_mesh.max()+self.x_mesh.min()) * 0.5
        mid_y = (self.y_mesh.max()+self.y_mesh.min()) * 0.5
        mid_z = (self.z_mesh.max()+self.z_mesh.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show()