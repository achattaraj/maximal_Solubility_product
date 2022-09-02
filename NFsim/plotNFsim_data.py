# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:49:26 2022

@author: Ani Chattaraj
"""

#from numpy import array, mean, log10, log2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 

font = {'family' : 'Arial',
        'size'   : 16}

plt.rc('font', **font)


def plotFreeConc(path, idx=[2,4], labels=['poly_A', 'poly_B']):
    # path: Location of the simulation 
    # m2cf: molecular count to concentration factor
    data = np.loadtxt(path + '/pyStat/Mean_Observable_Counts.txt')
    for i, elem in enumerate(idx):
        plt.plot(data[:,0]*1e3, data[:,elem], label = 'Free ' + labels[i])
        plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Molecular concentration ($\mu$M)')
    plt.show()

def plotClusterDistribution(path):
    df = pd.read_csv(path + '/pyStat/SteadyState_distribution.csv')
    cs, foTM = df['Cluster size'], df['foTM']
    aco = sum(cs*foTM)
    plt.bar(cs, foTM, fc='grey', ec='k')
    plt.axvline(aco, ls='dashed', color='k', lw=2, label=f'ACO = {aco:.2f} molecules')
    plt.legend()
    plt.xlabel('Cluster size (molecules)')
    plt.ylabel('Fraction of total molecules')
    plt.show()
    
        
    
    
    