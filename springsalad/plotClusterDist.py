# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:15:10 2021

@author: achattaraj
"""
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from numpy import log10

font = {'family' : 'Arial',
        'size'   : 20}

plt.rc('font', **font)

# file = "./test_dataset/A5_B5_flex_3nm_2nm_count_200_SIM_FOLDER/A5_B5_flex_3nm_2nm_count_200_SIM.txt"
# filepath = '/'.join(file.split('/')[:-1])

def plotDynamics(filepath, label=None):
    '''
    
    Parameters
    ----------
    filepath : String
        Location of the simulation 
    label : Boolen, optional
        Whether to put a title on the plot. The default is None.

    Returns
    -------
    Plot the average cluster occupnacy as a function of time

    '''
    df = pd.read_csv(filepath + '/pyStat/Cluster_stat/Clustering_dynamics.csv')
    plt.plot(df['Time (ms)'],df['ACO'],'ro--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Average cluster occupancy\n(molecules)')
    plt.title(label)
    plt.show()
    
def plotDistribution(filepath):
    '''

    Parameters
    ----------
    filepath : String
        Location of the simulation 

    Returns
    -------
    Plot the cluster distribution at steady state

    '''
    df = pd.read_csv(filepath + '/pyStat/Cluster_stat/SteadyState_distribution.csv')
    cs, fTM = df['Cluster size'], df['foTM']
    aco = sum(cs*fTM)
    plt.bar(cs, height=fTM, width=0.2, fc='grey', ec='k')
    plt.axvline(aco, ls='dashed', lw=1.5, color='k', label=f"ACO = {aco:.2f}")
    plt.xlabel('Cluster size (molecules)', labelpad=12)
    plt.ylabel('Fraction of total molecules', labelpad=14)
    plt.legend()
    plt.show()
 
    

