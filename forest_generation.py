from perlin_noise import PerlinNoise
from numpy.random import binomial

def get_perlin_noise_map(octaves: list, weights: list, width: int, height: int, seed: int = -1):
	"""
	Generates a perlin noise map

	Parameters:
		octaves: List of octaves that will used
		weights: The weights of each octave
		width: The width (number of columns) of the noise map
		height: The height (number of rows) of the noise map
		seed: Seed for noise generation. If negative no seed will be used (default: -1)

	Returns:
		Returns a 2D array with the given dimensions that store the noise value for each element
	"""
	# Checking lengths are the same
	if len(octaves) != len(weights):
		raise ValueError(f"The length of octaves and weights were not the same - octaves: {len(octaves)}, weights: {len(weights)}")
	
	# Generating perlin noise functions
	noise_funcs = list()
	for octave in octaves:
		if seed > 0:
			noise_funcs.append(PerlinNoise(octaves=octave, seed=seed))
		else:
			noise_funcs.append(PerlinNoise(octaves=octave))
	
	# Generating noise map
	map = list()
	for row in range(height):
		map.append(list())
		for column in range(width):
			noise_val = 0
			for func, weight in zip(noise_funcs, weights):
				noise_val += weight * func([column / width, row / height])
			map[-1].append(noise_val)

	return map

def generate_forest(height: int, width: int, seed: int = -1):
	"""
	Generates a forest

	Parameters:
		height: The height of the forest
		width: The width of the forest
		seed: Seed for random generation. If the seed is negative then no seed will be used. (default: -1)

	Returns:
		Returns the forest. The dimensions are (height x width)
	"""
	# Initialize matrix
	forest = [[dict() for y in range(width)] for x in range(height)]

	# Get perlin noise map
	noise_map = get_perlin_noise_map([3, 6, 12, 24], [1, 0.5, 0.25, 0.125], width, height, seed)

	# Populating forest matrix
	for row in range(height):
		for column in range(width):
			# Defining vegetation attribute
			if noise_map[row][column] < -0.2:
				forest[row][column]['vegetation'] = 0	# Nothing
				forest[row][column]['max_vegetation'] = 0
			elif noise_map[row][column] < -0.1:
				forest[row][column]['vegetation'] = 1	# Vegetation - 0
				forest[row][column]['max_vegetation'] = 1
			elif noise_map[row][column] < 0:
				forest[row][column]['vegetation'] = 2	# Vegetation - 1/4
				forest[row][column]['max_vegetation'] = 2
			elif noise_map[row][column] < 0.1:
				forest[row][column]['vegetation'] = 3	# Vegetation - 1/2
				forest[row][column]['max_vegetation'] = 3
			elif noise_map[row][column] < 0.3:
				forest[row][column]['vegetation'] = 4	# Vegetation - 3/4
				forest[row][column]['max_vegetation'] = 4
			else:
				forest[row][column]['vegetation'] = 5	# Vegetation - full
				forest[row][column]['max_vegetation'] = 5

			# Defining fire attribute
			forest[row][column]['fire'] = False

			# Defining charred atttribute
			forest[row][column]['charred'] = False

	return forest
