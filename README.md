# galaxy_analyzer 🌌

The main purpose of this package is to make plots of the top-down density gas and the longitude-velocity diagram of any galaxy (supposedly). Using the positional and velocity components, we could find:

$$ l = \arctan(\frac{y − y_{Ro}}{x − x_{Ro}}) $$ 

where, x and y are positional components from your galaxy sims; and $x_{Ro}$ and $y_{Ro}$ are observor's positional component.

$$ vlos = \sqrt{v_x^2 + v_y^2} \sin(l - \theta) - V_{obs} \sin(l) $$

where $v_x$ and $v_y$ are velocity component from your galaxy sims; $V_{obs}$ is the azimuthal velocity at the observer's position; and $\theta = \arctan(\frac{v_y}{v_x})$ 

# package installation 

You may start with `pip install galaxy_analyzer` and install it in your conda environment. 

You may also use git clone: `https://github.com/TowyZheng/galaxy_analyzer.git`. 

But, don't forget to: 
>`cd /PATH/TO/<<GALAXY_ANALYZER>>`\
>`git stash`\
>`git pull origin main`\
>`git stash pop` 
to get the latest version of our package.





