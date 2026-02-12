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
