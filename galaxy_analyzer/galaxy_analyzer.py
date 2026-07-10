import numpy as np
import matplotlib.pyplot as plt

class GalaxyAnalyzer:
    def __init__(self, rotating=False, degree=0.0, rotation_direction='clockwise', xRo=0.0, yRo=8.0, Vo=220.0):
        """
        Initializes the Galaxy Plotter.
        
        Parameters:
        - rotating (bool): Master toggle for whether the galaxy rotates.
        - degree (float): The angle to rotate the galaxy (in radians).
        - xRo, yRo (float): Coordinates of the observer (the Sun).
        - Vo (float): Velocity of the observer.
        """
        self.rotating = rotating
        self.degree = degree
        self.rotation_direction = rotation_direction.lower()
        self.xRo = xRo
        self.yRo = yRo
        self.Vo = Vo
        self.color_cycle = ['green', 'cyan', 'red', 'purple', 'blue', 'chocolate']

    # ---------------------------------------------------------
    # STATIC HELPER METHODS (Math that doesn't need class state)
    # ---------------------------------------------------------
    @staticmethod
    def rotator(x, y, theta, direction='clockwise'): 
        """
        Applies a 2D rotation matrix to the given coordinates.
        
        Parameter:
        - direction (boolean) = clockwise or anticlockwise
        """
        if direction == 'clockwise':
            xn = x * np.cos(theta) + y * np.sin(theta)
            yn = -x * np.sin(theta) + y * np.cos(theta)
        elif direction == 'anticlockwise':
            xn = x * np.cos(theta) - y * np.sin(theta)
            yn = x * np.sin(theta) + y * np.cos(theta)
        else:
            raise ValueError("rotation_direction must be 'clockwise' or 'anticlockwise'")
        return xn, yn

    @staticmethod
    def averager(x, y, fn):
        return fn(x), fn(y)

    @staticmethod
    def get_line_eq(x0, x1, y0, y1):
        return y0 - y1, x1 - x0, x0 * y1 - x1 * y0

    def color_fill_equation(self, spur_x_extracted, spur_y_extracted, x, y):
        equations = [
            self.get_line_eq(x0, x1, y0, y1) 
            for (x0, y0), (x1, y1) in zip(spur_x_extracted, spur_y_extracted)
        ]
        filter_mask = np.all([a*x + b*y + c > 0 for a, b, c in equations], axis=0)
        return filter_mask

    # ---------------------------------------------------------
    # CORE CLASS METHODS
    # ---------------------------------------------------------
    def rotate_galaxy(self, x, y, vx, vy, ax, s=0.01, alpha=0.065):
        """Rotates the galaxy coordinates and plots the initial gray point cloud."""
        theta = self.degree if self.rotating else 0.0

        new_x, new_y = self.rotator(x, y, theta, self.rotation_direction)
        new_vx, new_vy = self.rotator(vx, vy, theta, self.rotation_direction)

        ax.scatter(new_x, new_y, s=s, alpha=alpha)
        ax.set_aspect('equal', adjustable='box')
        
        return new_x, new_y, new_vx, new_vy

    # Note: make_points is responsible for both extracting the bounding box coordinates and plotting the outline if not rotating.
    def make_points(self, point_sets, ax, region_name):
        """Extracts and optionally rotates bounding box polygon coordinates."""
        coordinates = point_sets[region_name]
        
        # Separate x and y coordinates
        xs = coordinates[::2]
        ys = coordinates[1::2]

        theta = self.degree if self.rotating else 0.0
        
        pnts = []
        for x, y in zip(xs, ys):
            nx, ny = self.rotator(x, y, theta, self.rotation_direction)
            pnts.append([nx, ny])
            
        pnts.append(pnts[0]) # Close the loop
        pnts = np.array(pnts)

        # Plot bounding box outline if not rotating
        if not self.rotating:
            ax.plot(pnts[:, 0], pnts[:, 1], 'k--', lw=0.5)
            
        ax.set_aspect('equal', adjustable='box')
        return pnts

    # Note: extract_spur is responsible for both filtering points inside the polygon and plotting the highlighted spur with a yellow fill.
    # This function is needed to extract the positions and velocities of the extracted spur for the LV diagram later on.
    def extract_spur(self, x, y, vx, vy, m, ax, spur_extracted, color_index=0, s=0.01, alpha=0.08):
        """Filters points inside a polygon region and plots the highlighted spur."""
        filter0 = self.color_fill_equation(spur_extracted[:-1], spur_extracted[1:], x, y)

        cut_x = x[filter0]
        cut_y = y[filter0]
        cut_vx = vx[filter0]
        cut_vy = vy[filter0]
        mass_cut = m[filter0]

        colors = self.color_cycle[color_index % len(self.color_cycle)]

        print(f'Total mass in {colors} spur: {sum(mass_cut):.2e}')

        if self.rotating:
            print(f'xa_cut_for_lv = {len(cut_x)}')
            print(f'ya_cut_for_lv = {len(cut_y)}')

        # Plotting the highlighted particles and yellow polygon fill
        ax.scatter(cut_x, cut_y, s=s, c=colors, alpha=alpha, lw=2)
        ax.fill(spur_extracted[:, 0], spur_extracted[:, 1], fc='yellow', alpha=0.5, zorder=0)
        
        if not self.rotating:
            ax.plot(spur_extracted[:, 0], spur_extracted[:, 1], 'k--', lw=0.5, zorder=1)

        return cut_x, cut_y, cut_vx, cut_vy

    def initiate_lv(self, x, y, vx, vy, rhos):
        """Calculates longitude (l) and line-of-sight velocity (vLOS)."""
        alphav = np.arctan2(vy, vx)
        d = np.sqrt((x - self.xRo)**2 + (y - self.yRo)**2)

        l = np.arctan2((y - self.yRo), (x - self.xRo))
        
        # Coordinate wrap-around
        l[l >= np.pi / 2.0] = l[l >= np.pi / 2.0] - 2.0 * np.pi
        l[l < np.pi / 2.0] = l[l < np.pi / 2.0] + 0.5 * np.pi

        vLOS = (np.sqrt(vx**2 + vy**2) * np.sin(-alphav + l) - self.Vo * np.sin(l))
        
        # Note: I_approx is calculated but not returned in your original function. 
        I_approx = rhos / (d**2) 

        return alphav, l, vLOS

    def make_lv(self, l, vLOS, ax, first_plot=False, color_index=0, s=0.002, alpha=0.6):
        """Scatter plots the (l, vLOS) data onto the provided axes."""
        if first_plot:
            colors = 'grey'
        else:
            colors = self.color_cycle[color_index % len(self.color_cycle)]

        ax.scatter(l * 180. / np.pi, vLOS, s=s, c=colors, alpha=alpha)
        
        #ax.set_xlim(180, -180) # Inverted x-axis
        #ax.set_ylim(-250, 250)
        
        #ax.set_xlabel(r'$l \,\rm{[deg]}$', fontsize=20)  
        #ax.set_ylabel(r'$v_{LOS} \,\rm{[km/s]}$', fontsize=20) 
        #ax.tick_params(axis='both', labelsize=20, labelrotation=45)

        return l, vLOS