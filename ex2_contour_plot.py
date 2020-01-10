import numpy as np
import saddle
from saddle import rastrigin

# find k=1 saddle point of N=2 dimensional Rastrigin function around random position
tr = saddle.find_saddle(N=2,k=1,method='conv',trajectory=True,potential=rastrigin,verbose=True)

# plot contour and search trajectory of the algorithm
saddle.show_trajectory(tr,rastrigin)
