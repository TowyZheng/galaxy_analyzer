import numpy as np

class get_vRadial:
    """
    A utility to calculate the radial and azimuthal velocities of a galaxy 
    based on its position and velocity components.
    """
    def __init__(self, x, y, vx, vy):
        """
        Parameters:
        - x, y: Arrays representing the positions of the galaxy in Cartesian coordinates.
        - vx, vy: Arrays representing the velocity components of the galaxy in Cartesian coordinates.
        """
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.vx = np.array(vx).flatten()
        self.vy = np.array(vy).flatten()

    def get_vRadial(self):
        theta = np.arctan2(self.y, self.x) # Calculate angle in radians [-pi, pi]
        Theta_p = np.copy(theta) # Create a copy to keep original theta intact
        Theta_p[theta<0.] = theta[theta<0.]+2.*np.pi # Add 2*pi to negative values to map to [0, 2*pi]
        rcirc = np.sqrt(self.x*self.x + self.y*self.y)
        vcirc = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        
        Omega = vcirc / rcirc # Angular frequency
        
        vr = (self.vx*np.cos(Theta_p) + self.vy*np.sin(Theta_p)) # radial
        vp = -(self.vy*np.cos(Theta_p) - self.vx*np.sin(Theta_p)) # azimuthal
        return rcirc, vcirc, vr, vp, Omega