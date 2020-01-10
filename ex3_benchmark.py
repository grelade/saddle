import numpy as np
import saddle
from saddle import rastrigin
import timeit
import matplotlib.pyplot as plt
import pickle

# find local minimum of N=2 dimensional Rastrigin function around random position (default values)
#qsaddle = saddle.find_saddle(verbose=True)

Nkpairs = [[(5*i,k) for k in range(5*i+1)] for i in range(9,10)]

vals = {}
for Nrow in Nkpairs:
    x,y,yerr=[],[],[]
    if Nrow[0][0]>1:
        for row in Nrow:
            print('computing (N,k) pair =',row)
            x+=[row[1]];
            temp = timeit.repeat('saddle.find_saddle(N='+str(row[0])+',k='+str(row[1])+',alpha=0.000001,epsilon=1e-14)','import saddle',number=1,repeat=100);
            y+=[np.mean(temp)];
            yerr+=[np.std(temp)];

        vals[Nrow[0][0]] = [x,y,yerr]


with open('data.pkl','wb') as file:
    pickle.dump(vals,file)

for n,xy in vals.items():
    plt.errorbar(xy[0],xy[1],yerr=xy[2],label='N='+str(n))

plt.legend()
plt.show()

#tr = saddle.find_saddle(N=2,k=1,method='conv',trajectory=True,potential=rastrigin,verbose=True)

# print the point
#print('saddle:',np.transpose(qsaddle.numpy()))

#saddle.show_trajectory(tr,rastrigin)
#help(saddle.find_saddle)
#help(saddle.show_trajectory)
