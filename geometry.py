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

class NACA5:
    def __init__(self, naca_code):
        self.naca_code = naca_code
        # NACA 5-digit: first digit is design CL*100/15, 2nd-3rd are position of max camber, last 2 are thickness
        self.cl = int(self.naca_code[0]) * 0.15 / 100
        self.m = int(self.naca_code[1:3]) / 1000  # Position of maximum camber
        self.t = int(self.naca_code[3:]) / 100    # Maximum thickness
        # k1 coefficient for NACA 5-digit camber line
        self.k1 = 361.4 * self.cl
    
    def camber_line(self, x):
        f1 = self.cl/self.m**2*(2*self.m*x-x**2)
        f2 = self.cl/(1-self.m)**2*((1-2*self.m)+2*self.m*x-x**2)
        return f1*(x<self.m)+f2*(x>self.m)
    
    def thickness_distribution(self, x):
        thickness = 5*self.t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        return thickness