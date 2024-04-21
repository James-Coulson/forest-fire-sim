from copy import deepcopy
import itertools
from simulation import simulate_forest
from forest_generation import generate_forest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import log
import concurrent.futures as thr

"""
This file runs the forest simulations for varying conditions and then plots the output to the user.
It also makes use of threading so that the simulations can be run concurrently. This makes for a much faster simulation.

If you are afraid of threading use run_simulation.py instead.

Dimensions tested:	
	- lightning probability (gamma)
	- max lightning (zeta)
	- fire decay steps (sigma)
	- fire spread probability (mu)
	- Regrowth probability (delta)
"""

# Defining height and width
height = 100
width = 100

# Generating forest
forest = generate_forest(height, width)

# --------------------- Producing time series for differing a single variable --------------------- #

# Printing what is happening
print("\n!! ~~ Currently producing time series plot ~~ !!\n")

# Running simulations
with thr.ThreadPoolExecutor(max_workers=5) as executor:
	results = { executor.submit(simulate_forest, deepcopy(forest), height, width, initial_lightning=False, no_output=True, time_steps=400, regrowth_prob = x): x for x in np.arange(0, 1, 0.1) }
	for stats in thr.as_completed(results):
		prob = results[stats]
		plt.plot(stats.result()['time'], stats.result()['percent_charred'], label=f"Regrowth Probability = {round(prob, 4)}")

plt.suptitle("Percent Charred Time Series for Differing Regrowth Probabilities")
plt.xlabel("Time ($t$)")
plt.ylabel("Percent charred (%)")
plt.legend()
plt.show()

# --------------------- Determining distribution of percentage charred --------------------- #

# gamma 	= 0.03
# zeta 	= 3
# sigma 	= 3
# mu 		= 0.10
# delta 	= 0.14
 
# # Printing what is happening
# print("\n!! ~~ Making log distribution of percentage charred ~~ !!\n")

# # Running simulations
# stats = simulate_forest(deepcopy(forest), height, width, initial_lightning = False, no_output=True, time_steps=600, lightning_prob = gamma, max_lightning= zeta, fire_decay = sigma, fire_spread_prob = mu, regrowth_prob = delta)

# temp = list()

# for x in stats['percent_charred']:
# 	if x != 0:
# 		temp.append(log(x))

# plt.hist(temp, bins = 20)

# plt.suptitle(f"Distribution of log(percentage charred)")
# plt.xlabel("Percent charred (%)")
# plt.ylabel("Frequency")
# plt.show()

# --------------------- Producing 3D plot of differing environment values --------------------- #

# # Printing what is going on
# print("\n!! ~~ Currently producing 3D plot and time series plot ~~ !!\n")

# # Defining values to be simulated
# # Form: [lightning probability, max lightning, fire decay steps, fire spread probability, regrowth probability]
# gamma 	= np.linspace(0, 1, 3)
# zeta 	= [3]
# sigma 	= [3]
# mu 		= np.linspace(0, 1, 3)
# delta 	= np.linspace(0, 1, 3)

# # Printing parameters
# print(f"gamma: 	{gamma}")
# print(f"zeta: 	{zeta}")
# print(f"sigma:	{sigma}")
# print(f"mu:	{mu}")
# print(f"delta: 	{delta}")

# params = [gamma, zeta, sigma, mu, delta]
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# fig_pc = plt.figure()
# ax_pc = fig_pc.add_subplot()

# with thr.ThreadPoolExecutor(max_workers=20) as executor:
# 	results = dict()

# 	count = 1
# 	for x1, x2, x3, x4, x5 in itertools.product(*[gamma, zeta, sigma, mu, delta]):
# 		results[executor.submit(simulate_forest, deepcopy(forest), height, width, initial_lightning=False, no_output=True, time_steps=200, sim_num = count, lightning_prob=x1, max_lightning=x2, fire_decay=x3, fire_spread_prob=x4, regrowth_prob=x5)] = [x1, x2, x3, x4, x5]
# 		count += 1

# 	for stats in thr.as_completed(results):
# 		prob = results[stats]
# 		ax.scatter(prob[0], prob[3], prob[4], c=mpl.cm.get_cmap('autumn')(int(256*np.mean(stats.result()['percent_charred'][-100:]))))#color=np.mean(stats.result()['percent_charred'][-1]))
# 		ax_pc.plot(stats.result()['time'], stats.result()['percent_charred'], label=f"{prob}")
		
# 		print(f"{prob} : {np.mean(stats.result()['percent_charred'])}")
# 		# plt.plot(stats.result()['time'], stats.result()['percent_charred'], label=f"$\gamma$ = {prob[0]}")

# ax.set_xlabel("Lightning Probability")
# ax.set_ylabel("Fire Spread Probability")
# ax.set_zlabel("Regrowth Probability")

# plt.legend()
# plt.show()


# --------------------- Producing Parameter Space Manifold where Catastrophic Fires occur --------------------- #
#	This tends to crash from segmentation faults. This is a problem with the matplotlib library for 
# 	plotting and stroing large amounts of data.
#

# # Printing what is going on
# print("\n!! ~~ Currently producing 3D manifold plot ~~ !!\n")

# # Defining values to be simulated
# # Form: [lightning probability, max lightning, fire decay steps, fire spread probability, regrowth probability]
# gamma 	= np.linspace(0, 1, 4)
# zeta 	= [3]
# sigma 	= [3]
# mu 		= np.linspace(0, 1, 4)
# delta 	= np.linspace(0, 1, 4)

# # Printing parameters
# print(f"gamma: 	{gamma}")
# print(f"zeta: 	{zeta}")
# print(f"sigma:	{sigma}")
# print(f"mu:	{mu}")
# print(f"delta: 	{delta}")

# params = [gamma, zeta, sigma, mu, delta]
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# with thr.ThreadPoolExecutor(max_workers=20) as executor:
# 	results = dict()

# 	count = 1
# 	for x1, x2, x3, x4, x5 in itertools.product(*[gamma, zeta, sigma, mu, delta]):
# 		results[executor.submit(simulate_forest, deepcopy(forest), height, width, initial_lightning=False, no_output=True, time_steps=100, sim_num = count, lightning_prob=x1, max_lightning=x2, fire_decay=x3, fire_spread_prob=x4, regrowth_prob=x5)] = [x1, x2, x3, x4, x5]
# 		count += 1

# 	for stats in thr.as_completed(results):
# 		prob = results[stats]
# 		if np.mean(stats.result()['percent_charred'][-100:]) >= 0.7:
# 			ax.scatter(prob[0], prob[3], prob[4], c=mpl.cm.get_cmap('autumn')(int(256*np.mean(stats.result()['percent_charred'][-100:]))))
		
# 		print(f"{prob} : {np.mean(stats.result()['percent_charred'])}")
		

# ax.set_xlabel("Lightning Probability")
# ax.set_ylabel("Fire Spread Probability")
# ax.set_zlabel("Regrowth Probability")

# plt.legend()
# plt.show()