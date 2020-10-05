import numpy as np
import time, pickle as pkl 
import matplotlib.pyplot as plt
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *

trunk = 'fp-summary'

with open('4peaks.pkl', 'rb') as f:
    res = pkl.load(f)
    
    plt.figure()
    
    for algo in res:
        curves = res[algo]['curves']
        cv = get_curves(curves, algo)
    
        p50 = cv['p50']
        p33 = cv['p33']
        p66 = cv['p66']
        mean = np.array(cv['mean'])
        std = np.array(cv['std'])
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label=algo)
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.xlim([0,150])
    plt.savefig('{}-iter.pdf'.format(trunk))
    plt.show()    
    
    

        
        
