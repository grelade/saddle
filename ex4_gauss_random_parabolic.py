import math
import numpy as np
import saddle
from saddle import gauss_random,gauss_random_params,parabolic
import torch

# sample random parameters for the potential
n=2;
Kmax=20;
Domega = .5;

#random part parameters
params = gauss_random_params(Kmax,Domega,n);

#parabolic parameter
m=1.1;
mu=m*(2*math.pi)**(n/2);

# define auxilary potential parabolic + gauss_random
def f0(q):
    return parabolic(q,mu) + gauss_random(q,params)

#tr = saddle.find_saddle(N=n,k=0,potential=f0,method='conv',trajectory=True,alpha=0.1,epsilon=1e-10,verbose=True)
tr = torch.tensor([[0.],[0.]])
# plot contour and search trajectory of the algorithm
saddle.show_trajectory(tr,f0,resolution=20,region={-4,4})
