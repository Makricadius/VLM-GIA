from src.necessities.math_functions import deg, rad, rotate
from src.geometry.NACA_airfoils import NACA4
from src.geometry.paneles import Panels4
import numpy as np
import matplotlib.pyplot as plt


class trapezoidal_wing:
    def __init__(self, surface_area=1, aspect_ratio=1, taper_ratio=1, twist=0, sweep=0, airfoil=None):
        """
        This function calculates the geometric parameters of the wing based on the input parameters: 
         - surface area (S)
         - aspect ratio (A)
         - taper ratio (taper_ratio)
         - twist (twist)
         - sweep angle (sweep)
        """
        self.S = surface_area
        self.A = aspect_ratio
        self.taper_ratio = taper_ratio
        self.twist = rad(twist) # tip twist
        self.sweep = rad(sweep) # sweep at c/4
        self.airfoil = airfoil
        if airfoil == None:
            """If no airfoil is specified, a simple flat plate is used as default, which is equivalent to a NACA 0100 with 0% thickness."""
            self.airfoil = NACA4("0100")
    
    def calculate_wing_parameters(self):
        self.b = np.sqrt(self.A*self.S)
        self.cgm = self.b/self.A
        self.cr = 2*self.cgm/(1+self.taper_ratio)
        self.ct = self.cr*self.taper_ratio
        self.cam = 2/3*self.cr*(1+self.taper_ratio+self.taper_ratio**2)/(1+self.taper_ratio)
        self.yca = self.b/6*(1+2*self.taper_ratio)/(1+self.taper_ratio)
        self.xca = 0.25*self.cr+np.tan(self.sweep)*self.yca
    
    def mesh(self, Nb=20, Nc=1, mesh_type_span="uniform", mesh_type_chord="uniform", simetric=True):
        self.simetric = simetric
        """Span-wise meshing"""
        self.Nb = Nb
        """Defining the mesh function spacer for span-wise. 
        x is a value between -1 and 1 for simetrical wings, and between 0 and 1 for non-simetrical wings."""
        if mesh_type_span == "uniform":
            self.mesh_function_span = lambda x: x
        elif mesh_type_span == "cosine":
            self.mesh_function_span = lambda x: -np.cos((x+1)*np.pi/2)
        else:
            self.mesh_function_span = lambda x: mesh_type_span(x)
        
        """Chord-wise meshing"""
        self.Nc = Nc
        """Defining the mesh function spacer for chord-wise.
        x is a value between 0 and 1."""
        if mesh_type_chord == "uniform":
            self.mesh_function_chord = lambda x: x
        elif mesh_type_chord == "cosine":
            self.mesh_function_chord = lambda x: (1-np.cos(x*np.pi))/2
        else:
            self.mesh_function_chord = lambda x: mesh_type_chord(x)
        
        """Meshing the wing"""
        """Defining the nodes span-wise"""
        if self.simetric:
            y_nondim = self.mesh_function_span(np.linspace(-1, 1, Nb+1))
        else:
            y_nondim = self.mesh_function_span(np.linspace(0, 1, Nb+1))
        y_nodes = y_nondim*self.b/2
        y_scaling = 2*abs(y_nodes)/self.b

        """Defining the nodes coord-wise"""
        x_nondim = self.mesh_function_chord(np.linspace(0, 1, Nc+1))
        x_coord_nodes = np.array([x_nondim]).T[::-1]
        x_coord = self.cr + (self.ct-self.cr)*y_scaling
        """Defining the nodes curvature-wise"""
        z_coord = self.airfoil.camber_line(x_nondim)
        """Generating X & Y & Z meshes for the nodes"""
        mesh = np.ones((Nc+1,Nb+1))
        x_mesh = (x_coord_nodes*mesh-0.25)*x_coord # placing c/4 at y=0 so that when twisting the wing, the twist is applied at c/4, and applying tapering
        y_mesh = mesh*y_nodes # dimensional span-wise coordinates
        z_mesh = np.array([z_coord]).T[::-1]*mesh
        """"Applying the twist at c/4 to the nodes"""
        self.linear_torsion = self.twist*y_scaling
        x_mesh, z_mesh = rotate(x_mesh, z_mesh, -self.linear_torsion)
        """Applying the sweep to the nodes"""
        x_mesh += np.tan(self.sweep)*abs(y_nodes)
        """"Applying the dihedral angle to the nodes"""
        # y_mesh, z_mesh = rotate(y_mesh, z_mesh, self.diedro*np.sign(y_nodes))
        """Leading edge of root coord at 0,0,0,"""
        x_mesh += 0.25*self.cr # placing back the leading edge of the root at x=0, and thus the rest of the wing is placed accordingly
        """Storing the nodes in the class"""
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh
        
        """Reshaping 2D matrices to 1D arrays to create the panels"""
        xA = np.reshape(self.x_mesh[:-1,:-1], Nb*Nc)
        xB = np.reshape(self.x_mesh[:-1,1:], Nb*Nc)
        xC = np.reshape(self.x_mesh[1:,:-1], Nb*Nc)
        xD = np.reshape(self.x_mesh[1:,1:], Nb*Nc)
        yA = np.reshape(self.y_mesh[:-1,:-1], Nb*Nc)
        yB = np.reshape(self.y_mesh[:-1,1:], Nb*Nc)
        yC = np.reshape(self.y_mesh[1:,:-1], Nb*Nc)
        yD = np.reshape(self.y_mesh[1:,1:], Nb*Nc)
        zA = np.reshape(self.z_mesh[:-1,:-1], Nb*Nc)
        zB = np.reshape(self.z_mesh[:-1,1:], Nb*Nc)
        zC = np.reshape(self.z_mesh[1:,:-1], Nb*Nc)
        zD = np.reshape(self.z_mesh[1:,1:], Nb*Nc)

        self.panels = Panels4(xA, yA, zA, xB, yB, zB, xC, yC, zC, xD, yD, zD) # class to easyly pass all panels created to the VLM solver
    
    def print_parameters(self):
        print(" Carácteristicas geométricas:")
        print("  S =", self.S)
        print("  alargamiento =", self.A)
        print("  estrechamiento =", self.estrechamiento)
        print("  flecha =", deg(self.flecha),"º")
        print("  torsión =", deg(self.torsion),"º")
        print("  Envergadura =", self.b)
        print("  Cuerda geometrica media =", self.cgm)
        print("  Cuerda encastre =", self.cr)
        print("  Cuerda punta =", self.ct)
        print("  Cuerda aerodinamica media =", self.cam)
        print("  X centro aerodinamico =", self.xca)
        print("  Y centro aerodinamico =", self.yca)
        print("  Inclinación de cuerdas:")
        for e in [0, 0.25, 0.5, 0.75, 1]:
            inclination = np.arctan(np.tan(self.flecha)+2*self.cr/self.b*(1-self.estrechamiento*np.cos(self.torsion))*(0.25-e))
            print(f"   Al {round(e*100):>3}% cuerda = {round(deg(inclination),2)} º = {round(inclination,2)} rad")
        
    def plot_nodes(self, scale=[1,1,1]):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(self.x_mesh.shape[0]):
            ax.plot(self.x_mesh[i, :]*scale[0], self.y_mesh[i, :]*scale[1], self.z_mesh[i, :]*scale[2], 'b-', alpha=0.6)
        
        for j in range(self.x_mesh.shape[1]):
            ax.plot(self.x_mesh[:, j]*scale[0], self.y_mesh[:, j]*scale[1], self.z_mesh[:, j]*scale[2], 'b-', alpha=0.6)
        
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