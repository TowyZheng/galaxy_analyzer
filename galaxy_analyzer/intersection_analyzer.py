import numpy as np

class IntersectionAnalyzer:
    """
    A mathematical utility to calculate the intersections 
    and angles between a complex curve and a linear dataset.
    """
    def __init__(self, x1, y1, x2, y2, resolution=10000):
        """
        Parameters:
        - x1, y1: Arrays representing the first dataset (the curve).
        - x2, y2: Arrays representing the second dataset (the line).
        - resolution: The number of points to use when creating the common x-axis for interpolation.
        """
        
        # Flatten inputs to guarantee 1D arrays
        self.x1 = np.array(x1).flatten()
        self.y1 = np.array(y1).flatten()
        self.x2 = np.array(x2).flatten()
        self.y2 = np.array(y2).flatten()
        self.resolution = resolution
        
        # Initialize result variables as None until calculate() is run
        self.has_intersection = False
        self.x_int = None
        self.y_int = None
        self.m1 = None
        self.m2 = None
        self.angle_deg = None
        self.common_x = None
        self.y1_common = None
        self.y2_common = None

    def get_arrays_to_plot(self):
        return self.x1, self.y1, self.x2, self.y2
    
    def calculate(self):
        """Executes parametric intersection math, ignoring loops and sorting."""
        
        # 1. Fit linear regression to Dataset 2
        self.m2, c_reg = np.polyfit(self.x2, self.y2, 1)
        
        # 2. We calculate the vertical distance from every point on the spiral arm 
        # curve to the regression line.
        # Distance D = y_curve - (m * x_curve + c)
        self.D = self.y1 - (self.m2 * self.x1 + c_reg)
        
        # 3. An intersection occurs wherever the curve crosses the line 
        # (D changes sign)
        crossings = np.where(np.diff(np.sign(self.D)))[0]
        if len(crossings) == 0:
            print("Notice: The parametric curve never crosses the line, no intersection found.")
            return False
            
        # 4. If multiple intersections
        # We find the intersection physically closest to the center of Dataset 2
        center_x2 = np.mean(self.x2)
        center_y2 = np.mean(self.y2)
        
        best_idx = crossings[0]
        min_dist = float('inf') # Positive infinity
        
        for idx in crossings:
            dist = np.sqrt((self.x1[idx] - center_x2)**2 + (self.y1[idx] - center_y2)**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                
        # 5. Linear interpolation on that specific tiny segment to find exact crossing coords
        x_a, y_a = self.x1[best_idx], self.y1[best_idx]
        x_b, y_b = self.x1[best_idx+1], self.y1[best_idx+1]
        
        dx = x_b - x_a
        dy = y_b - y_a
        
        if dx == 0: # Vertical segment safety
            self.x_int = x_a
            self.y_int = self.m2 * self.x_int + c_reg
            self.m1 = np.inf
        else:
            self.m1 = dy / dx # Local slope of the spiral arm at the crossing
            
            if self.m1 == self.m2:
                self.x_int, self.y_int = x_a, y_a
            else:
                self.x_int = (c_reg + self.m1 * x_a - y_a) / (self.m1 - self.m2)
                self.y_int = self.m2 * self.x_int + c_reg

        """# 6. Calculate Angle
        denom = 1 + self.m1 * self.m2
        
        # Handle perpendicular or infinite slopes
        if np.isinf(self.m1) or np.isinf(self.m2) or np.isclose(denom, 0):
            print("Warning: Slopes are perpendicular (or infinite), angle is 90 degrees.")
            self.angle_deg = 90.0
            
        # Handle parallel slopes
        elif self.m1 == self.m2:
            print("Notice: Slopes are parallel, angle is 0 degrees.")
            self.angle_deg = 0.0
            
        # Standard angle calculation
        else:
            angle_rad = np.arctan(np.abs((self.m1 - self.m2) / denom))
            self.angle_deg = np.degrees(angle_rad)
            
        # Unconditionally flag the intersection as True since we made it this far
        self.has_intersection = True"""
        
        # 6. Calculate Angle (Using Vector Cosine Method)
        
        # Handle cases where one or both lines are perfectly vertical (infinite slope)
        if np.isinf(self.m1) and np.isinf(self.m2):
            print("Notice: Both lines are vertical (parallel).")
            self.angle_deg = 0.0
            
        elif np.isinf(self.m1):
            # If line 1 is vertical, its direction vector is <0, 1>
            # The angle is just 90 degrees minus the angle of line 2
            self.angle_deg = np.degrees(np.abs(np.pi/2 - np.arctan(self.m2)))
            
        elif np.isinf(self.m2):
            # If line 2 is vertical
            self.angle_deg = np.degrees(np.abs(np.pi/2 - np.arctan(self.m1)))
            
        # Standard cosine calculation for all normal slopes
        else:
            dot_product = np.abs(1 + self.m1 * self.m2)
            mag1 = np.sqrt(1 + self.m1**2)
            mag2 = np.sqrt(1 + self.m2**2)
            
            # Calculate cos(theta)
            cos_theta = dot_product / (mag1 * mag2)
            
            # np.clip acts as a safety net against floating-point rounding errors
            # (e.g., stopping cos_theta from becoming 1.0000000000000002 which crashes arccos)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_theta)
            self.angle_deg = np.degrees(angle_rad)
            
        # Unconditionally flag the intersection as True since we made it this far
        self.has_intersection = True

        return True, self.x_int, self.y_int, self.m1, self.m2, self.angle_deg, self.common_x, self.y1_common, self.y2_common
