# Wing class - represents basic wing geometry
class Wing:
    """Represents a wing with a specific type/profile."""
    def __init__(self, type):
        """Initialize a wing with a given type.
        
        Args:
            type (str): The type or profile of the wing
        """
        self.type = type

# NACA 4-digit airfoil class
class NACA4:
    """Represents a 4-digit NACA airfoil profile.
    
    NACA 4-digit format: MPXX
    - M: Maximum camber as fraction of chord (first digit / 100)
    - P: Position of maximum camber (second digit / 10)
    - XX: Maximum thickness (last two digits / 100)
    """
    def __init__(self, naca_code):
        """Initialize NACA 4-digit airfoil.
        
        Args:
            naca_code (str): 4-digit NACA code (e.g., '2412')
        """
        self.naca_code = naca_code
        # Extract parameters from NACA code
        self.f = int(self.naca_code[0])/100     # Maximum camber
        self.xf = int(self.naca_code[1])/10     # Position of maximum camber
        self.t = int(self.naca_code[2:])/100    # Maximum thickness
    
    def camber_line(self, x):
        """Calculate the camber line (mean line) at position x.
        
        Uses piecewise quadratic formulas for positions before and after
        the maximum camber point.
        
        Args:
            x (float): Position along chord (0 to 1)
            
        Returns:
            float: Camber line ordinate
        """
        # Camber line before maximum camber point
        f1 = self.f/self.xf**2*(2*self.xf*x-x**2)
        # Camber line after maximum camber point
        f2 = self.f/(1-self.xf)**2*((1-2*self.xf)+2*self.xf*x-x**2)
        # Select appropriate formula based on position
        return f1*(x<self.xf)+f2*(x>self.xf)
    
    def thickness_distribution(self, x):
        """Calculate the thickness distribution at position x.
        
        Uses the standard NACA thickness formula with a smooth trailing edge.
        
        Args:
            x (float): Position along chord (0 to 1)
            
        Returns:
            float: Half-thickness value
        """
        # Standard NACA thickness formula
        thickness = 5*self.t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        # Smooth trailing edge with interpolation
        return thickness*(1-x)**0.01

# NACA 5-digit airfoil class
class NACA5:
    """Represents a 5-digit NACA airfoil profile.
    
    NACA 5-digit format: LPMXX
    - L: Design lift coefficient (first digit * 0.15 / 100)
    - PM: Position of maximum camber (digits 2-3 / 1000)
    - XX: Maximum thickness (last two digits / 100)
    """
    def __init__(self, naca_code):
        """Initialize NACA 5-digit airfoil.
        
        Args:
            naca_code (str): 5-digit NACA code (e.g., '23012')
        """
        self.naca_code = naca_code
        # Extract parameters from NACA code
        self.cl = int(self.naca_code[0]) * 0.15 / 100    # Design lift coefficient
        self.m = int(self.naca_code[1:3]) / 1000         # Position of maximum camber
        self.t = int(self.naca_code[3:]) / 100           # Maximum thickness
        # k1 coefficient for NACA 5-digit camber line
        self.k1 = 361.4 * self.cl
    
    def camber_line(self, x):
        """Calculate the camber line (mean line) at position x.
        
        Uses piecewise quadratic formulas adapted from NACA 4-digit,
        but with NACA 5-digit coefficients.
        
        Args:
            x (float): Position along chord (0 to 1)
            
        Returns:
            float: Camber line ordinate
        """
        # Camber line before maximum camber point
        f1 = self.cl/self.m**2*(2*self.m*x-x**2)
        # Camber line after maximum camber point
        f2 = self.cl/(1-self.m)**2*((1-2*self.m)+2*self.m*x-x**2)
        # Select appropriate formula based on position
        return f1*(x<self.m)+f2*(x>self.m)
    
    def thickness_distribution(self, x):
        """Calculate the thickness distribution at position x.
        
        Uses the standard NACA thickness formula.
        
        Args:
            x (float): Position along chord (0 to 1)
            
        Returns:
            float: Half-thickness value
        """
        # Standard NACA thickness formula
        thickness = 5*self.t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        return thickness