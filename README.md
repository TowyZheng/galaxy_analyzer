# galaxy_analyzer 🌌

The main purpose of this package is to make plots of the top-down density gas and the longitude-velocity diagram of any galaxy (supposedly). Using the positional and velocity components, we could find:

$$ l = \arctan(\frac{y − y_{Ro}}{x − x_{Ro}}) $$ 

where, x and y are positional components from your galaxy sims; and $x_{Ro}$ and $y_{Ro}$ are observor's positional component.

$$ v_{los} = \sqrt{v_x^2 + v_y^2} \sin(l - \theta) - V_{obs} \sin(l) $$

where $v_x$ and $v_y$ are velocity component from your galaxy sims; $V_{obs}$ is the azimuthal velocity at the observer's position; and $\theta = \arctan(\frac{v_y}{v_x})$ 

# package installation 

You may start with `pip install galaxy_analyzer` and install it in your conda environment. 

You may also use git clone: `https://github.com/TowyZheng/galaxy_analyzer.git`. 

If you go with the git clone method, don't forget to do: 
>`cd /PATH/TO/<<GALAXY_ANALYZER>>`\
>`git stash`\
>`git pull origin main`\
>`git stash pop` 

to get the latest version of our package.

If you are interested in exploring all the module functions, you can visit here to see them in detail: https://galaxy-analyzer.readthedocs.io/en/latest/galaxy_analyzer.html

# After pip installed...

After you've pip installed the package, you can start by importing it into your .py or .ipynb files. 
To ensure that you have the correct version of the package, run the following cell so that you don't miss our latest version.

```python
import importlib.metadata
import galaxy_analyzer

print("Imported from:", galaxy_analyzer.__file__)
print("Installed version:", importlib.metadata.version("galaxy-analyzer"))

# Sanity check: make sure this resolved to site-packages, not the local repo checkout.
assert "site-packages" in galaxy_analyzer.__file__, (
    "galaxy_analyzer resolved to a local path, not the installed package — "
    "check you aren't accidentally shadowing it with a local folder."
)
```

## 1. `GalaxyAnalyzer` — rotation + plotting 

This function is our main focus. We use this package to plot top-down views and LV diagrams of our galaxy sims. 
The following is an example of how you could call our package through pip installation.

```python
import numpy as np
import matplotlib.pyplot as plt
from galaxy_analyzer.galaxy_analyzer import GalaxyAnalyzer

Ro   = 8.0  # Observer's positional location
Vo   = 220  # Observer's rotational velocity at that location
"exc."
lobs = 0.  # the line-of-sight longitude angle is set 0 from the observer's location to the galactic center 
#change coordinate system to one centered on the Sun rather than centre of the galaxy:
yRo  = Ro * np.sin((lobs+90.)*np.pi/180.)
xRo  = Ro * np.cos((lobs+90.)*np.pi/180.)

analyzer = GalaxyAnalyzer(rotating=False, rotation_direction='anticlockwise', xRo=xRo, yRo=yRo, Vo=-Vo)

fig, ax = plt.subplots(figsize=(6, 6))
new_x, new_y, new_vx, new_vy = analyzer.rotate_galaxy(x, y, vx, vy, ax, s=0.5, alpha=0.05) 
ax.set_title("Synthetic Milky Way-like disk")
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
plt.show()
```
![top-down galaxy plot](/figs/topdown_plot.png)

```python
fig1, ax1 = plt.subplots(figsize=(10, 4))
alpha, longitude, vLOS = analyzer.initiate_lv(new_x, new_y, new_vx, new_vy, density)
analyzer.make_lv(longitude, vLOS, ax1)
ax1.set_title("LV Diagram of Synthetic Milky Way-like disk")
ax1.set_ylabel(r"$v_{los}$ [km/s]")
ax1.set_xlabel('longitude angle [deg]')
plt.show()
```
![lv galaxy plot](/figs/lv_plot.png)

## 2. `get_vRadial` — radial / azimuthal velocities

This function can be when you want to plot the circular velocity map, radial velocity map, and frequency plot of your galaxy sims. 
The following example shows how we use it to plot the circular velocity map.

```python
from galaxy_analyzer.get_vRadial import get_vRadial

vrad_calc = get_vRadial(x, y, vx, vy)
rcirc, vcirc, vr, vp, Omega = vrad_calc.get_vRadial()

print(f"Median galactocentric radius: {np.median(rcirc):.2f} kpc")
# Sign follows this class's internal rotation convention; magnitude is what matters here.
print(f"Median azimuthal speed: {np.median(np.abs(vp)):.1f} km/s (expect ~200-230 km/s for a MW-like disk)")

plt.scatter(rcirc, -vp, color='k',s=0.1,alpha=0.2, label = '$V_c$')
#plt.plot(Ro, Vo, 'ro', label = 'Sun', markersize = 10)

plt.legend(loc=4)
plt.xlabel('R [kpc]')
plt.ylabel('V [km/s]')

plt.show()
```
![circular_velocity plot](/figs/circularVelocity.png)

## 3. `IntersectionAnalyzer` — curve/line intersection

We use this package to calculate the intersection point and the cosine of the angle between the tangent line of the synthetic logarithmic spiral arm and the linear regression line of the spur. 
The purpose can also be to calculate the intersection point and the angles of intersection of a complex curve and a linear line. 
You may follow an example here: 

```python
from galaxy_analyzer.intersection_analyzer import IntersectionAnalyzer

# A synthetic spiral-arm curve
theta = np.linspace(0, 2 * np.pi, 500)
arm_x = (2 + 0.3 * theta) * np.cos(theta)
arm_y = (2 + 0.3 * theta) * np.sin(theta)

# A straight line-of-sight crossing through the arm
line_x = np.linspace(-10, 10, 100)
line_y = 0.5 * line_x + 1.0

int_analyzer = IntersectionAnalyzer(arm_x, arm_y, line_x, line_y)
int_analyzer.calculate()

print("Intersection found:", int_analyzer.has_intersection)
print(f"Intersection point: ({int_analyzer.x_int:.2f}, {int_analyzer.y_int:.2f})")
print(f"Crossing angle: {int_analyzer.angle_deg:.1f} deg")
```

## 4. `FigureSaver` — saving output

This module allows you to save figures as PNG files, combine figures to make a GIF video, combine two or more GIF videos side-by-side, and make GIF videos using PNG files previously saved in a folder. 
The following example will show you how we use this module to save a plot as a PNG file.

```python
from galaxy_analyzer.plot_utilities import FigureSaver

saver = FigureSaver(base_directory=".")
saver.save(folder_name="test_output", file_name="synthetic_galaxy_plot", fig=fig, dpi=150)
```



