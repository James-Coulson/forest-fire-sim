from copy import deepcopy
from simulation import simulate_forest
from forest_generation import generate_forest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import concurrent.futures as thr

height = 100
width = 100

# ---------------------------------------------------------------------------------------------------
#	Dimensions:
#		- Lightning probability
#		- Maxmimum number of lightning
#		- Fire spread base probability
# 		- 
#


forest = generate_forest(height, width)

stats = simulate_forest(forest, height, width, no_output=False, draw_wind=True, time_steps=1000, manual_increment=False, plot_frequency=1, regrowth_prob=0.1, initial_lightning=True)

plt.plot(stats.result()['time'], stats.result()['percent_charred'], label=f"lighting probability = {prob}")
plt.legend()
plt.show()