from typing import Callable
from plotting import FOREST_COLORMAP, FOREST_COLORMAP_NORM, draw_forest
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import linspace, meshgrid, sqrt
import numpy as np
from numpy.random import binomial
from constants import *
from random import randint
import random as rd
from utility import clamp, get_number_of_trees
from math import ceil

# --------------------------------------- Example Wind Functions --------------------------------------- #

def no_wind(width: int, height: int, time: int):
	"""
	This is an example wind function where the wind direction and magntude is zero over the forest. 
	Note that it is required that the wind values can only be between -1 and 1, also note that the vertical direction (v)
		are inverted in the return statement. This is due to the graphing of the vector field being inverted in the y axis.

	The vector field this wind function creates is:
	
		F(x) = 0 i + 0 j

	Parameters:
		width: The width of the forest
		height: The height of the forest
		time: The current time

	Returns:
		A list that can be passed to the draw_forest function as well as used by simulate_forest
	"""
	column, row = meshgrid(linspace(0, width - 1, width), linspace(0, height - 1, height))
	u = np.full(column.shape, 0)
	v = np.full(column.shape, 0)
	return [column, row, u, -v]


def uniform_constant_wind(width: int, height: int, time: int):
	"""
	This is an example wind function where the wind direction and magntude is uniform over the forest. 
	Note that it is required that the wind values can only be between -1 and 1, also note that the vertical direction (v)
		are inverted in the return statement. This is due to the graphing of the vector field being inverted in the y axis.

	The vector field this wind function creates is:
	
		F(x) = 0.5 i + 0.5 j

	Parameters:
		width: The width of the forest
		height: The height of the forest
		time: The current time

	Returns:
		A list that can be passed to the draw_forest function as well as used by simulate_forest
	"""
	column, row = meshgrid(linspace(0, width - 1, width), linspace(0, height - 1, height))
	u = np.full(column.shape, 0.5)
	v = np.full(column.shape, 0.5)
	return [column, row, u, -v]

def outward_wind(width: int, height: int, time: int):
	"""
	This is an example wind function where the wind direction and magntude constantly points out from (0, 0). 
	Note that it is required that the wind values can only be between -1 and 1.

	The vector field in this wind function is
	
		F(x) = x / sqrt(x^2 + y^2) i + y / sqrt(x^2 + y^2) j

	Parameters:
		width: The width of the forest
		height: The height of the forest
		time: The current time

	Returns:
		A list that can be passed to the draw_forest function as well as used by simulate_forest
	"""
	column, row = meshgrid(linspace(0, width - 1, width), linspace(0, height - 1, height))
	u = (column - 50)/(2 * np.sqrt((column - 50)**2 + (row - 50)**2))
	v = (row - 50)/(2 * np.sqrt((column - 50)**2 + (row - 50)**2))
	return [column, row, u, -v]

def centered_outward_wind(width: int, height: int, time: int):
	"""
	This is an example wind function where the wind direction and magntude constantly points out from (50, 50). 
	Note that it is required that the wind values can only be between -1 and 1.

	The vector field in this wind function is
	
		F(x) = x / sqrt(x^2 + y^2) i + y / sqrt(x^2 + y^2) j

	Parameters:
		width: The width of the forest
		height: The height of the forest
		time: The current time

	Returns:
		A list that can be passed to the draw_forest function as well as used by simulate_forest
	"""
	column, row = meshgrid(linspace(0, width - 1, width), linspace(0, height - 1, height))
	u = column/(2 * np.sqrt(column**2 + row**2))
	v = row/(2 * np.sqrt(column**2 + row**2))
	return [column, row, u, -v]

def centered_spiral_wind(width: int, height: int, time: int):
	"""
	This is an example wind function where the wind direction and magntude is a sprial centered aroun (50, 50). 
	Note that it is required that the wind values can only be between -1 and 1.

	The vector field in this wind function is
	
		F(x) = x / sqrt(x^2 + y^2) i - y / sqrt(x^2 + y^2) j

	Parameters:
		width: The width of the forest
		height: The height of the forest
		time: The current time

	Returns:
		A list that can be passed to the draw_forest function as well as used by simulate_forest
	"""
	column, row = meshgrid(linspace(0, width - 1, width), linspace(0, height - 1, height))
	column
	u = (row - 50)/(2 * np.sqrt((column- 50)**2 + (row - 50)**2))
	v = -(column- 50)/(2 * np.sqrt((column- 50)**2 + (row - 50)**2))
	return [column, row, u, -v]


# --------------------------------------- Forest Simulation Functions --------------------------------------- #

def simulate_forest(forest: list, height: int, width: int, time_steps: int = 100, wind_func: Callable = outward_wind, plot_frequency: int = 0, draw_wind: bool = True, 
					manual_increment: bool = False, lightning_prob: float = 0.05, max_lightning: int = 3, initial_lightning: bool = True, seed: int = -1, no_output: bool = False,
					fire_decay: int = FIRE_DECAY_TIME_STEPS, fire_spread_prob: float = FIRE_SPREAD_BASE_PROBABILITY, regrowth_prob: float = VEGETATION_REGROWTH_PROBABILITY, sim_num: int = -1):
	"""
	Runs the forest fire simulation for the given forest

	Parameters:
		forest: The forest that the simulation will run on
		height: The height (number of rows) of the forest
		width: The width (number of columns) of the forest
		time_steps: The number of time_steps to simulate (default: 100)
		wind_func: Callable used to generate the wind vector field at each time step
		plot_frequency: Number of time steps between plot calls, 0 means the graph will not render (default: 0)
		draw_wind: If True the wind vector field will be plotted over the forest. (default: True)
		manual_increment: If True the simulation will block between time steps and the user can manually increment the simulation (default: False)
		lightning_prob: Probability that a single lighning strike occurs in a time step (default: 0.01)
		max_lightning: The maximum number of lightning strikes that can happen per round (default: 3)
		initial_lightning: Whether there should be a guaranteed lightning at the beginning of the simulation (default: True)
		seed: Seed for random generation. If the seed is negative then no seed will be used. (default: -1)
		no_output: If only the dataframe should be returned and no other output
		fire_decay: The number of time steps that a fire lasts. (default: FIRE_DECAY_TIME_STEPS)
		fire_spread_prob: The base probability that a fire will spread. (default: FIRE_SPREAD_BASE_PROBABILITY)
		regrowth_prob: The base probability that a cell will regrow. (default: VEGETATION_REGROWTH_PROBABILITY)
		sim_num: A positive number that can be displayed before running the simulation (default: -1)

	Returns:
		stats: dictionary of lists that contain different statistics for the simultion
	"""
	print(f"{sim_num} - Simulating for: lightning_prob:{lightning_prob}, max_lightning:{max_lightning}, fire_decay:{fire_decay}, fire_spread_prob:{fire_spread_prob}, regrowth_prob:{regrowth_prob}")

	# Setting seed
	if seed > 0:
		np.random.seed(seed)
		rd.seed(seed)

	# Create initial lightning strike
	if initial_lightning:
		while True:
			row = randint(0, height - 1)
			column = randint(0, width - 1)

			if forest[row][column]['vegetation']:
				forest[row][column]['fire'] = fire_decay
				break

	# Defining constants
	NUM_TREES = get_number_of_trees(forest, width, height)

	# Defining statistics dictionary
	stats = { "time": list(), "percent_charred": list() }

	# Perform time steps
	for t in range(time_steps):
		# Calculating new wind field
		wind_field = wind_func(width, height, t)

		# Plotting forest
		if plot_frequency != 0 and t % plot_frequency == 0 and not no_output:
			draw_forest(forest, width, height, wind_field, draw_wind=draw_wind, block=False)

		# If the user wants to manually increment time
		if manual_increment:
			input()

		# Getting copy of forest
		new_forest = deepcopy(forest)

		# --- Stats --- #
		stats['time'].append(t)
		percent_charred = 0.0
		# ------- #

		# Performing simulation time increment
		for row in range(height):
			for column in range(width):
				# --- Vegetation regrowth --- #
				# Count number of vegetative neighbours
				# count = 1 if forest[max(0, row - 1)][column]['vegetation'] else 0			# Up
				# count += 1 if forest[min(row + 1, height - 1)][column]['vegetation'] else 0	# Down
				# count += 1 if forest[row][max(0, column - 1)]['vegetation'] else 0			# Left
				# count += 1 if forest[row][min(column + 1, width - 1)]['vegetation'] else 0	# Right
				count = forest[max(0, row - 1)][column]['vegetation']
				count += forest[min(row + 1, height - 1)][column]['vegetation']
				count += forest[row][max(0, column - 1)]['vegetation']
				count += forest[row][min(column + 1, width - 1)]['vegetation']

				# Spreading fire
				if forest[row][column]['fire']:
					# Upwards spread
					if forest[max(0, row - 1)][column]['vegetation'] and not forest[max(0, row - 1)][column]['charred'] and not forest[max(0, row - 1)][column]['fire']: 
						# Calculting upwards spread probability
						up_prob = clamp(fire_spread_prob * forest[max(0, row - 1)][column]['vegetation'] * (1 + wind_field[3][max(0, row - 1)][column]), 0, 1)

						# Spreading fire
						if binomial(1, up_prob):
							new_forest[max(0, row - 1)][column]['fire'] = fire_decay				# Up

					# Downwards spread
					if forest[min(row + 1, height - 1)][column]['vegetation'] and not forest[min(row + 1, height - 1)][column]['charred'] and not forest[min(row + 1, height - 1)][column]['fire']:
						# Calculating downwards spread probability
						down_prob = clamp(fire_spread_prob * forest[min(row + 1, height - 1)][column]['vegetation'] * (1 - wind_field[3][min(row + 1, height - 1)][column]), 0, 1)
						
						# Spreading fire
						if binomial(1, down_prob):
							new_forest[min(row + 1, height - 1)][column]['fire'] = fire_decay

					# Left Spread
					if forest[row][max(0, column - 1)]['vegetation'] and not forest[row][max(0, column - 1)]['charred'] and not forest[row][max(0, column - 1)]['fire']: 
						# Calcualting probability of spreading left
						left_prob = clamp(fire_spread_prob * forest[row][max(0, column - 1)]['vegetation'] * (1 - wind_field[2][row][max(0, column - 1)]), 0, 1)
						
						# Spreading fire
						if binomial(1, left_prob):
							new_forest[row][max(0, column - 1)]['fire'] = fire_decay 

					# Right Spread
					if forest[row][min(column + 1, width - 1)]['vegetation'] and not forest[row][min(column + 1, width - 1)]['charred'] and not forest[row][min(column + 1, width - 1)]['fire']:
						# Calculating probability of spreading right
						right_prob = clamp(fire_spread_prob * forest[row][min(column + 1, width - 1)]['vegetation'] * (1 + wind_field[2][row][min(column + 1, width - 1)]), 0, 1)

						# Spreading fire
						if binomial(1, right_prob):
							new_forest[row][min(column + 1, width - 1)]['fire'] = fire_decay 
					
					# Putting out fire, setting charred and vegetation
					if new_forest[row][column]['fire'] == 1:
						new_forest[row][column]['charred'] = True
						new_forest[row][column]['fire'] -= 1
					else:
						new_forest[row][column]['fire'] -= 1
					new_forest[row][column]['vegetation'] = 0

				elif binomial(1, clamp(regrowth_prob * count / 4.0, 0, 1)) and (new_forest[row][column]['charred'] or new_forest[row][column]['vegetation'] > 0) and (new_forest[row][column]['vegetation'] < new_forest[row][column]['max_vegetation']):
					# Stop charred
					if new_forest[row][column]['charred']:
						new_forest[row][column]['charred'] = False 
					
					# Increment vegetation
					new_forest[row][column]['vegetation'] = min(new_forest[row][column]['vegetation'] + ceil(1 * count / 4.0), 5)

				# Collecting stats
				if new_forest[row][column]['charred']:
					percent_charred += 1.0 / (NUM_TREES)
	
		
		# --- Stats --- #
		stats['percent_charred'].append(percent_charred)

		# Lightning strikes
		# Note: here if the strike happens at a square with not vegetation
		# 	no fire will occur. This is realistic and so there is no need to ensure
		# 	that a fire will result.
		strikes = binomial(max_lightning, lightning_prob)
		for strike in range(strikes):
			row = randint(0, height - 1)
			column = randint(0, width - 1)

			if forest[row][column]['vegetation'] and not forest[row][column]['charred']:
				new_forest[row][column]['fire'] = fire_decay


		# Updating forest
		forest = new_forest

	# Block at the end of the process
	if not no_output:
		plt.show()
	
	return stats

