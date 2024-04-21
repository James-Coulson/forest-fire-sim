
def clamp(n, minn, maxn):
	"""
	This function is used to clamp the vlaue of n within the specified minimum and maximum

	Paramters:
		n: Number to be clamped
		minn: Minimum bound of n
		maxn: Maximum bound of n

	Returns:
		The clamped value of n
	"""
	return max(min(maxn, n), minn)

def get_number_of_trees(forest: list, width: int, height: int) -> int:
	"""
	This function is used to count the number of vegetative squares in the forest

	Parameters:
		forest: The forest

	Returns:
		The number of vegetative squares
	"""
	n = 0
	for row in range(height):
		for column in range(width):
			if forest[row][column]['vegetation']:
				n += 1
	
	return n
