from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from numpy import array, meshgrid, linspace
import numpy as np

# Defining colors used in colormap
FOREST_COLORMAP = ListedColormap([
	'#0087FF',		# Nothing
	'#E6EC36',		# Vegetation - 0
	'#BEEC36',		# Vegetation - 1/4
	'#96EC36',		# Vegetation - 1/2
	'#6EEC36',		# Vegetation - 3/4
	'#46EC36',		# Vegetation - Full
	'#FF0000',		# Fire
	'#20201E'		# Charred
])

# Bounds of colors in colormap
FOREST_COLORMAP_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

# Norm used to plot
FOREST_COLORMAP_NORM = BoundaryNorm(FOREST_COLORMAP_BOUNDS, FOREST_COLORMAP.N)


def get_colormap(forest: list, width: int, height: int):
	"""
	This function converts the given forest array to a color mapping of the array

	Parameters:
		forest: 2D matrix of dictionaries
		width: The width of the forest
		height: The height of the forest

	Returns:
		A 2D color mapping
	"""
	ret = []
	for row in range(height):
		ret.append([])
		for column in range(width):
			if forest[row][column]['fire']:
				ret[-1].append(6)
			elif forest[row][column]['charred']:
				ret[-1].append(8)
			else:
				ret[-1].append(forest[row][column]['vegetation'])

	return ret

def draw_forest(forest: list, width: int, height: int, wind_field: list = list(), block: bool = True,
				draw_wind: bool = True):
	"""
	Draws a the goven forest

	Parameters:
		forest: 2D matrix of dictionaries.
		width: The width of the forest.
		height: The height of the forest.
		wind_field: List returned from the wind_func callable. If an empty list is passed no wind will be plotted (default: list())
		block: If True the function will block the thread once forest has been drawn. (default: True)
		draw_wind: If True the wind vector filed will be plotted over the forest. (default: True)
	"""
	
	# Plotting forest
	plt.imshow(array(get_colormap(forest, width, height)), cmap=FOREST_COLORMAP, norm=FOREST_COLORMAP_NORM)
	
	# Plotting wind
	if len(wind_field) == 4 and draw_wind:
		plt.quiver(wind_field[0], wind_field[1], wind_field[2], wind_field[3])

	# Drawing forest
	if not block:
		plt.draw()
		plt.pause(0.01)
		plt.clf()		# Used to clear previous drawings
	else:
		plt.show()