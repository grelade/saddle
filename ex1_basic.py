import numpy as np
import saddle
from saddle import rastrigin,ackley

# find saddle point of index k=5 with N=20 dimensional Ackley function around random position
qsaddle = saddle.find_saddle(N=20,k=5,potential=rastrigin,method='step',steps=1000,alpha=0.001)

# print out the saddle point
print('saddle point:',np.transpose(qsaddle.numpy()))
