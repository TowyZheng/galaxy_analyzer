import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import io
from PIL import Image

# Local Imports
from galaxy_analysis.galactic_plotter import GalacticPlotter

# Global Plotting Params
params = {"font.family": "serif", "mathtext.fontset": "stix"}
matplotlib.rcParams.update(params)
mapper = GalacticPlotter(rotating=False, degree=0.0, rotation_direction='clockwise', xRo=0.0, yRo=8.0, Vo=220.0)

class SyntheticPlotter(GalacticPlotter):
    def __init__(self, rotating=False, degree=0.0, rotation_direction='clockwise', xRo=0.0, yRo=8.0, Vo=220.0):
        """
        Initializes the Synthetic Galaxy Plotter.
        
        Parameters:
        - rotating (bool): Master toggle for whether the galaxy rotates.
        - degree (float): The angle to rotate the galaxy (in radians).
        - rotation_direction (str): 'clockwise' or 'counterclockwise' rotation (see the Galaxy structure).
        - xRo, yRo (float): Coordinates of the observer (the Sun).
        - Vo (float): Velocity of the observer.
        """
        # 1. Use super() to trigger the Parent's setup automatically
        super().__init__(rotating, degree, rotation_direction, xRo, yRo, Vo)
        
        # State Variables
        self.rotating = rotating
        self.degree = degree
        self.rotation_direction = rotation_direction.lower()
        self.xRo = xRo
        self.yRo = yRo
        self.Vo = Vo
        self.color_cycle = ['green', 'cyan', 'red', 'purple', 'blue', 'chocolate']
        
        # Physical Constants
        self.gcode = 6.674e-11 # Gravitational constant in m^3 kg^-1 s^-2
        self.kpc   = 3.0856e16 * 1000.0 # in meters
        self.Mo    = 1.99e30 # Solar mass in kg
        self.Mo10  = self.Mo * 1e10 # 10^10 solar masses
        self.Rc    = 0.1 # Core radius for the Dobbs curve in kpc

    
    #------------------ ARM ROUTINES (UPDATED FOR ROTATION)
    def logspiralarm(self, R, pa, Rs, aoff, c1, lsa, lwa, vmodel, ax_XY, ax_LV, RAngle=0.0):
        """
        This is a pure circular arm with no expansion velocity. 
        The rotation of the galaxy is handled by the RAngle parameter, which applies a rotation to the arm coordinates before plotting.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - Rs (float): Scale radius for the spiral arm in kpc.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - c1 (str): Color for plotting the arm.
        - lsa (str): Line style for plotting the arm.
        - lwa (float): Line width for plotting the arm.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - ax_LV (matplotlib axis): Axis for plotting the LV diagram.
        -> For ax_XY and ax_LV, the axis should already be set up with appropriate limits and labels before calling this function.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        g = (1.0 / (np.tan(pa))) * np.log((R / Rs))
        
        xl = R * np.cos(g + aoff)
        yl = R * np.sin(g + aoff)
        
        xl, yl = mapper.rotator(xl, yl, RAngle, self.rotation_direction)
        
        ln, = ax_XY.plot(xl, yl, linestyle=lsa, linewidth=lwa, c=c1)
        ln.set_solid_capstyle('round')

        vl = self.RCobs(R, vmodel, 'n')

        l = np.arctan2((yl - self.yRo), xl)
        l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
        l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi
        
        vLOS = (vl / np.sqrt(xl**2 + yl**2) - self.Vo / self.yRo) * self.yRo * np.sin(l)
        self.drawlv(l * 180. / np.pi, vLOS, ax_LV, lsa, lwa, c1)
        
    #------------------ ARM ROUTINES (UPDATED FOR ROTATION)
    def cropped_logspiralarm(self, R, pa, Rs, aoff, yc_max, yc_min, segment, c1, lsa, lwa, vmodel, ax_XY, ax_LV, RAngle=0.0):
        """
        This is a cropped version of a pure circular arm with no expansion velocity. 
        The rotation of the galaxy is handled by the RAngle parameter, which applies a rotation to the arm coordinates before plotting.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - Rs (float): Scale radius for the spiral arm in kpc.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - yc_max (float): Maximum y-coordinate for cropping the arm in kpc.
        - yc_min (float): Minimum y-coordinate for cropping the arm in kpc.
        - segment (int): Index of the segment to plot after cropping. 
        -> For example, if the arm is cropped into 3 segments based on the y-range, segment=0 will plot the first segment, segment=1 will plot the second segment, and so on.
        - c1 (str): Color for plotting the arm.
        - lsa (str): Line style for plotting the arm.
        - lwa (float): Line width for plotting the arm.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - ax_LV (matplotlib axis): Axis for plotting the LV diagram.
        -> For ax_XY and ax_LV, the axis should already be set up with appropriate limits and labels before calling this function.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        g = (1.0 / (np.tan(pa))) * np.log((R / Rs))
        
        xl = R * np.cos(g + aoff)
        yl = R * np.sin(g + aoff)
        
        # Create a global mask to filter out points outside the specified y-range before rotation
        # This is a boolean mask
        global_mask = (yl >= yc_min) & (yl <= yc_max)
        
        # A difference of 1 means False -> True (start of a segment)
        # A difference of -1 means True -> False (end of a segment)
        
        changes = np.diff(global_mask.astype(int))
        
        starts = np.where(changes == 1)[0] + 1 
        ends = np.where(changes == -1)[0] + 1
        
        # Edge case handling: if the first point is True, we need to add a start at index 0
        if global_mask[0]:
            starts = np.insert(starts, 0, 0)
        if global_mask[-1]:
            ends = np.append(ends, len(global_mask))
            
        # Choose your segment here
        # first occurance = 0, second occurance = 1, etc.
        segment_choice = segment
        
        if segment_choice < len(starts):
            s_idx = starts[segment_choice]
            e_idx = ends[segment_choice]
            
            # Create the final, isolated arrays
            x_isolated = xl[s_idx:e_idx]
            y_isolated = yl[s_idx:e_idx]
            R_isolated = R[s_idx:e_idx]
        
            x_isolated, y_isolated = mapper.rotator(x_isolated, y_isolated, RAngle, self.rotation_direction)
            
            ln, = ax_XY.plot(x_isolated, y_isolated, linestyle=lsa, linewidth=lwa, c=c1)
            ln.set_solid_capstyle('round')

            vl = self.RCobs(R_isolated, vmodel, 'n')

            l = np.arctan2((y_isolated - self.yRo), x_isolated)
            l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
            l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi
            
            vLOS = (vl / np.sqrt(x_isolated**2 + y_isolated**2) - self.Vo / self.yRo) * self.yRo * np.sin(l)
            self.drawlv(l * 180. / np.pi, vLOS, ax_LV, lsa, lwa, c1)
        else:
            print(f"Error: Segment {segment_choice} doesn't exist! There are only {len(starts)} occurances.")
        
    def get_logspiral_lv_data(self, R, pa, Rs, aoff, vmodel, RAngle=0.0):
        """
        This is just to get the LV data for a pure circular arm with no expansion velocity.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - Rs (float): Scale radius for the spiral arm in kpc.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        g = (1.0 / (np.tan(pa))) * np.log((R / Rs))
        
        xl = R * np.cos(g + aoff)
        yl = R * np.sin(g + aoff)
        
        xl, yl = self.rotator(xl, yl, RAngle, self.rotation_direction)
        vl = self.RCobs(R, vmodel, 'n') 

        l = np.arctan2((yl - self.yRo), xl)
        l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
        l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi
        
        vLOS = (vl / np.sqrt(xl**2 + yl**2) - self.Vo / self.yRo) * self.yRo * np.sin(l)
        l_deg = l * 180. / np.pi
        
        return l_deg, vLOS
    
    def get_cropped_logspiralarm(self, R, pa, Rs, aoff, yc_max, yc_min, segment, vmodel, RAngle=0.0):
        """
        This is a cropped version of a pure circular arm with no expansion velocity. 
        The rotation of the galaxy is handled by the RAngle parameter, which applies a rotation to the arm coordinates before plotting.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - Rs (float): Scale radius for the spiral arm in kpc.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - yc_max (float): Maximum y-coordinate for cropping the arm in kpc.
        - yc_min (float): Minimum y-coordinate for cropping the arm in kpc.
        - segment (int): Index of the segment to plot after cropping. 
        -> For example, if the arm is cropped into 3 segments based on the y-range, segment=0 will plot the first segment, segment=1 will plot the second segment, and so on.
        - c1 (str): Color for plotting the arm.
        - lsa (str): Line style for plotting the arm.
        - lwa (float): Line width for plotting the arm.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - ax_LV (matplotlib axis): Axis for plotting the LV diagram.
        -> For ax_XY and ax_LV, the axis should already be set up with appropriate limits and labels before calling this function.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        g = (1.0 / (np.tan(pa))) * np.log((R / Rs))
        
        xl = R * np.cos(g + aoff)
        yl = R * np.sin(g + aoff)
        
        # Create a global mask to filter out points outside the specified y-range before rotation
        # This is a boolean mask
        global_mask = (yl >= yc_min) & (yl <= yc_max)
        
        # A difference of 1 means False -> True (start of a segment)
        # A difference of -1 means True -> False (end of a segment)
        
        changes = np.diff(global_mask.astype(int))
        
        starts = np.where(changes == 1)[0] + 1 
        ends = np.where(changes == -1)[0] + 1
        
        # Edge case handling: if the first point is True, we need to add a start at index 0
        if global_mask[0]:
            starts = np.insert(starts, 0, 0)
        if global_mask[-1]:
            ends = np.append(ends, len(global_mask))
            
        # Choose your segment here
        # first occurance = 0, second occurance = 1, etc.
        segment_choice = segment
        
        if segment_choice < len(starts):
            s_idx = starts[segment_choice]
            e_idx = ends[segment_choice]
            
            # Create the final, isolated arrays
            x_isolated = xl[s_idx:e_idx]
            y_isolated = yl[s_idx:e_idx]
            R_isolated = R[s_idx:e_idx]
        
            x_isolated, y_isolated = mapper.rotator(x_isolated, y_isolated, RAngle, self.rotation_direction)

            vl = self.RCobs(R_isolated, vmodel, 'n')

            l = np.arctan2((y_isolated - self.yRo), x_isolated)
            l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
            l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi
            
            vLOS = (vl / np.sqrt(x_isolated**2 + y_isolated**2) - self.Vo / self.yRo) * self.yRo * np.sin(l)
            l_deg = l * 180. / np.pi
        else:
            print(f"Error: Segment {segment_choice} doesn't exist! There are only {len(starts)} occurances.")
        return l_deg, vLOS

    def expandingarm(self, R, pa, aoff, c1, lsa, lwa, vmodel, vb, ax_XY, ax_LV, RAngle=0.0):
        """
        This is an expanding arm where the expansion velocity (vb) is added to the rotational velocity. 
        The rotation of the galaxy is handled by the RAngle parameter, which applies a rotation to the arm coordinates before plotting.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - c1 (str): Color for plotting the arm.
        - lsa (str): Line style for plotting the arm.
        - lwa (float): Line width for plotting the arm.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - vb (float): Expansion velocity of the arm in km/s.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - ax_LV (matplotlib axis): Axis for plotting the LV diagram.
        -> For ax_XY and ax_LV, the axis should already be set up with appropriate limits and labels before calling this function.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        Rs   = np.exp(-aoff * np.tan(pa)) # Calculate Rs based on the angular offset and pitch angle. Rs is the radius at which the arm crosses the positive x-axis.
        g    = (1.0 / (np.tan(pa))) * np.log((R / Rs)) 
        
        xl = R * np.cos(g)
        yl = R * np.sin(g)
        
        xl, yl = self.rotator(xl, yl, RAngle, self.rotation_direction)

        ln, = ax_XY.plot(xl, yl, linestyle=lsa, linewidth=lwa, c=c1)
        ln.set_solid_capstyle('round')

        vl = self.RCobs(R, vmodel, 'n')

        l = np.arctan2((yl - self.yRo), xl)
        l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
        l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi

        vx     = +vl * np.sin(g) + vb * np.cos(g)
        vy     = -vl * np.cos(g) + vb * np.sin(g)
        alphav = np.arctan2(vy, vx)

        vLOS = (np.sqrt(vx**2 + vy**2) * np.sin(-alphav + l) - self.Vo * np.sin(l))
        self.drawlv(l * 180. / np.pi, vLOS, ax_LV, lsa, lwa, c1)
        
    def get_expanding_arm_data(self, R, pa, aoff, c1, lsa, lwa, vmodel, vb, ax_XY, ax_LV, RAngle=0.0):
        """
        This is just to get the LV data for an expanding arm where the expansion velocity (vb) is added to the rotational velocity.
        
        Parameters:
        - R (array): Radial distances for the arm.
        - pa (float): Pitch angle of the spiral arm in degrees.
        - aoff (float): Angular offset for the spiral arm in degrees.
        - c1 (str): Color for plotting the arm.
        - lsa (str): Line style for plotting the arm.
        - lwa (float): Line width for plotting the arm.
        - vmodel (str): Rotation curve model to use for velocity calculations.
        - vb (float): Expansion velocity of the arm in km/s.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - ax_LV (matplotlib axis): Axis for plotting the LV diagram.
        -> For ax_XY and ax_LV, the axis should already be set up with appropriate limits and labels before calling this function.
        - RAngle (float): Additional rotation angle for the arm in radians.
        """
        pa   = pa * np.pi / 180.0
        aoff = aoff * np.pi / 180.0
        Rs   = np.exp(-aoff * np.tan(pa)) # Calculate Rs based on the angular offset and pitch angle. Rs is the radius at which the arm crosses the positive x-axis.
        g    = (1.0 / (np.tan(pa))) * np.log((R / Rs)) 
        
        xl = R * np.cos(g)
        yl = R * np.sin(g)
        
        xl, yl = self.rotator(xl, yl, RAngle, self.rotation_direction)
        vl = self.RCobs(R, vmodel, 'n')

        l = np.arctan2((yl - self.yRo), xl)
        l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2. * np.pi
        l[l < np.pi / 2.0]  = l[l < np.pi / 2.0]  + 0.5 * np.pi

        vx     = +vl * np.sin(g) + vb * np.cos(g)
        vy     = -vl * np.cos(g) + vb * np.sin(g)
        alphav = np.arctan2(vy, vx)

        vLOS = (np.sqrt(vx**2 + vy**2) * np.sin(-alphav + l) - self.Vo * np.sin(l))
        l_deg = l * 180. / np.pi
        return l_deg, vLOS

    def drawlv(self, l, v, Ax, lsa, lwa, c1):
        """
        Parameters:
        - l (array): Galactic longitude values in degrees.
        - v (array): Line-of-sight velocity values in km/s.
        - Ax (matplotlib axis): Axis for plotting the LV diagram.
        - lsa (str): Line style for plotting the LV data.
        - lwa (float): Line width for plotting the LV data.
        - c1 (str): Color for plotting the LV data.
        """
        l_diff = np.abs(np.diff(l, prepend=l[0]))
        l_masked = np.ma.masked_where(l_diff > 180, l)
        v_masked = np.ma.masked_where(l_diff > 180, v)
        
        ln, = Ax.plot(l_masked, v_masked, linestyle=lsa, linewidth=lwa, c=c1, alpha=0.5)
        ln.set_solid_capstyle('round')

    def barcartoon(self, length, width, angle, c1, ax_XY, RAngle=0.0):
        """
        Parameters:
        - length (float): Length of the bar in kpc.
        - width (float): Width of the bar in kpc.
        - angle (float): Angle of the bar with respect to the line connecting the Sun and the Galactic Center in degrees.
        - c1 (str): Color for plotting the bar.
        - ax_XY (matplotlib axis): Axis for plotting the XY plane (top-down view).
        - RAngle (float): Additional rotation angle for the bar in radians.
        """
        total_angle = -(angle + RAngle * 180.0 / np.pi)
        ax_XY.add_artist(Ellipse((0., 0.), 2*width, 2*length, angle=total_angle, facecolor=c1, edgecolor='w', zorder=100))

    #------------------ ROTATION CURVES
    def RCobs(self, R, CurveType, plotit):
        """
        This is the rotation curve from Pettitt2014, which includes contributions from the bulge, disk, and halo.
        
        Parameters:
        - R (array): Radial distances in kpc.
        - CurveType (str): 'obs', 'Binney', or 'Dobbs' to specify which rotation curve model to use.
            #'Binney' = strong log plateau
            #'Dobbs' = Clare's flat RC
            #'obs' = bulge+disc+halo potentials from Pettitt2014
        - plotit (str): 'y' to plot the individual components and total rotation curve, 'n' to skip plotting.
        """
        x = R
        y = np.zeros(len(x))
        z = 0.
        
        if plotit == 'y':
            plt.figure(20)

        if CurveType == 'obs':
            discpot, discv = self.MNdisc(x, y, z, 'XY')
            bulpot, bulv   = self.PlummerBulge(x, y, z, 'XY')
            halopot, halov = self.AllenHalo(x, y, z, 'XY')
            vtot = np.sqrt(np.power(discv, 2) + np.power(bulv, 2) + np.power(halov, 2))
            if plotit == 'y':
                plt.plot(R, discv)
                plt.plot(R, bulv)
                plt.plot(R, halov)
        elif CurveType == 'Binney':
            discpot, discv = self.Logdisc(x, y, z, 'XY')
            vtot = np.sqrt(np.power(discv, 2))
        elif CurveType == 'Dobbs':
            vtot = self.Vo * R * R / (self.Rc * self.Rc + R * R)

        if plotit == 'y':
            plt.plot(R, vtot, 'k--', linewidth=3, label='Total')
            plt.ylim(0, 300)
            plt.xlim(0, 14)
            plt.xlabel('R [kpc]')
            plt.ylabel('V [km/s]')
        return vtot

    def Logdisc(self, xi, yi, zi, Oflag):
        """
        This is the logarithmic disc potential from Binney & Tremaine, which produces a strong log plateau in the rotation curve.
        
        Parameters:
        - xi, yi, zi (array): Coordinates in kpc.
        - We are not going to use this Oflag (str): 'XY' for plotting the potential in the XY plane, 'LV' for plotting in the LV diagram (not used in this function but included for consistency with other potential functions).
        """
        Vc = 220000.0
        Co = (Vc**2) * 0.5
        Rc = 0.1 * self.kpc
        dzq = 1.0 / 0.7
        xi = xi * self.kpc
        yi = yi * self.kpc
        zi = zi * self.kpc
        r = np.sqrt(xi*xi + yi*yi)
        d2 = r*r
        r2term = ((Rc)**2 + d2 + (zi*dzq)**2)
        dr2term = 1. / r2term
        phi = -Co * np.log(r2term / (1 * self.kpc**2))  
        fr = -2. * Co * r * dr2term
        vr = np.sqrt(-fr * r) / 1000.
        return phi / 1e6, vr

    def MNdisc(self, xi, yi, zi, Oflag):
        """
        This is the Miyamoto-Nagai disc potential, which produces a more realistic rotation curve with a gradual rise and flattening at larger radii.
        
        Parameters:
        - xi, yi, zi (array): Coordinates in kpc.
        - Oflag (str): 'XY' for plotting the potential in the XY plane, 'LV' for plotting in the LV diagram (not used in this function but included for consistency with other potential functions).
        """
        Mdisk = 8.56 * self.Mo10
        adisk = 5.3 * self.kpc
        bdisk = 0.25 * self.kpc
        x = xi * self.kpc
        y = yi * self.kpc
        z = zi * self.kpc
        r = np.sqrt(x*x + y*y)
        d2 = r*r
        diskterm = np.sqrt(z*z + bdisk*bdisk)
        diskterm2 = np.sqrt(d2 + (adisk + diskterm)*(adisk + diskterm))
        phi = -self.gcode * Mdisk / diskterm2
        fr  = -r * self.gcode * Mdisk / (diskterm2*diskterm2*diskterm2)
        vr  = np.sqrt(-fr*r) / 1000.0 
        return phi / 1e6, vr

    def PlummerBulge(self, xi, yi, zi, Oflag):
        """
        This is the Plummer sphere potential for the bulge, which produces a steep rise in the rotation curve at small radii due to the concentrated mass of the bulge.
        
        Parameters:
        - xi, yi, zi (array): Coordinates in kpc.
        - Oflag (str): 'XY' for plotting the potential in the XY plane, 'LV' for plotting in the LV diagram (not used in this function but included for consistency with
        """
        Rbulge = 0.3873 * self.kpc
        Mbulge = 1.4 * self.Mo10
        x = xi * self.kpc
        y = yi * self.kpc
        z = zi * self.kpc
        r = np.sqrt(x*x + y*y + z*z)
        r2term = np.sqrt(r*r + Rbulge*Rbulge)
        phi = -self.gcode * Mbulge / r2term
        fextxi = -x * self.gcode * Mbulge / (r2term*r2term*r2term)
        fextyi = -y * self.gcode * Mbulge / (r2term*r2term*r2term)
        fextzi = -z * self.gcode * Mbulge / (r2term*r2term*r2term)
        vr = np.sqrt(np.sqrt(fextxi**2 + fextyi**2 + fextzi**2) * r) / 1000.
        return phi / 1e6, vr

    def AllenHalo(self, xi, yi, zi, Oflag):
        """
        This is the Allen & Santillan halo potential, which produces a gradual rise in the rotation curve at large radii due to the extended mass distribution of the dark matter halo. 
        The potential is designed to match observed rotation curves and includes a cutoff radius to prevent unphysical behavior at very large distances.
        
        Parameters:
        - xi, yi, zi (array): Coordinates in kpc.
        - Oflag (str): 'XY' for plotting the potential in the XY plane, 'LV' for plotting in the LV diagram (not used in this function but included for consistency with other potential functions).
        """
        Mhalo = 10.7 * self.Mo10
        rhalo = 12.0 * self.kpc
        rlim = 100.0 * self.kpc
        gamma = 2.02
        x = xi * self.kpc
        y = yi * self.kpc
        z = zi * self.kpc
        r = np.sqrt(x*x + y*y + z*z)
        r[r <= 0.] = 0.0001
        dr = 1.0 / r
        haloterm = ((r / rhalo)**1.02)
        Massinhalo = Mhalo * haloterm * (r / rhalo) / (1.0 + haloterm)     
        phi = (-self.gcode * Massinhalo / r - self.gcode * Mhalo / (1.02 * rhalo) * (-1.02 / (1.0 + (rlim / rhalo)**1.02) + np.log(1.0 + (rlim / rhalo)**1.02) + 1.02 / (1.0 + haloterm) - np.log(1.0 + haloterm)))
        haloforce = -self.gcode * Massinhalo * dr * dr
        fr = haloforce
        vr = np.sqrt(-fr * r) / 1000.0
        return phi / 1e6, vr