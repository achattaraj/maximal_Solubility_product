# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:45:57 2020

@author: achattaraj
"""
import re, sys, ast, os
import numpy as np
from collections import defaultdict, OrderedDict
from glob import glob
import matplotlib.pyplot as plt
from csv import writer
from numpy import array, mean

def ProgressBar(jobName, progress, length=40):
    completionIndex = round(progress*length)
    msg = "\r{} : [{}] {}%".format(jobName, "*"*completionIndex + "-"*(length-completionIndex), round(progress*100))
    if progress >= 1: msg += "\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    
    
class BoundFraction_NFsim:
    
    def __init__(self, inpath, binding_sites=[4,4], molecules=[]):
        self.inpath = inpath
        self.bs = np.array(binding_sites)
        self.mols = molecules
    
    def __repr__(self):
        simfile = self.inpath.split('/')[-1]
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}"
        return info
    
    def calc_BF(self, spfile):
        csList, bfList = [], []
        MCL_list = [] # MCL: molecular cross link (number of bonds per molecule)
        
        with open(spfile, 'r') as tf:
            lines = tf.readlines()[2:]
            for line in lines:
                if not (re.search('Sink', line) or re.search('Source', line)):
                    cs = line.count('.') + 1 # cluster size
                    if cs > 1:
                        cc = int(line.split()[-1]) # cluster count
                        bound_sites = line.count('!')
                        molCounts = np.array([line.count(mol) for mol in self.mols])
                        total_sites = sum(molCounts* self.bs)
                        bf = bound_sites/ total_sites # bound fraction
                        csList.extend([cs]*cc)
                        bfList.extend([bf]*cc)
                        
                        species = line.split()[0].split('.')
                        get_bc = lambda s: s.count('!') # bc: bond count
                        MCL = [get_bc(sp) for sp in species] 
                        # molecular cross link
                        MCL_list.extend(MCL)
                           
        return csList, bfList, MCL_list
    
    def get_boundFraction(self, saveIt=True):
        csList, bfList = [], []
        MCL_stat = []
        
        spfiles = glob(self.inpath + "/*.species")[:]
        N = len(spfiles)
        print(f"Processing {N} species files ...\n")
        
        for i, spfile in enumerate(spfiles):
            ProgressBar("Progress", (i+1)/N)
            cs, bf, mcl = self.calc_BF(spfile)
            csList.extend(cs)
            bfList.extend(bf)
            MCL_stat.extend(mcl)
        
        counts_norm = {k: (MCL_stat.count(k)/len(MCL_stat)) for k in set(MCL_stat)}
        self.plotBondsPerMolecule(counts_norm)    
        if saveIt:
            path = self.inpath + '/pyStat'
            if not os.path.isdir(path):
                os.makedirs(path)
            np.save(path + '/BF_stat.npy', array([csList,bfList]))
            with open(path + '/Bonds_per_single_molecule.csv', "w", newline='') as of:
                obj = writer(of)
                obj.writerow(['BondCounts','frequency'])
                obj.writerows(zip(counts_norm.keys(), counts_norm.values()))
    
    
    @staticmethod
    def plotBondsPerMolecule(countDict):
        fig, ax = plt.subplots(figsize=(5,3))
        bonds, freq = countDict.keys(), countDict.values()
        ax.bar(bonds, freq, width=0.3, color='grey')
        ax.set_xlabel('Bonds per molecule')
        ax.set_ylabel('Frequency')
        plt.show()
    
    def plotBFpattern(self, label=None):
        data = np.load(self.inpath + '/pyStat/BF_stat.npy')
        cs, bf = data[0], data[1]
        
        plt.figure(figsize=(5,3))
        
        plt.scatter(np.log10(cs), bf, c='grey', s=16)
        m_bf = mean(bf[cs>100])
        
        plt.axhline(m_bf, ls='dashed', lw=1, color='k',
                    label = f'mean bf = {m_bf:.2f}\n(clusters > 100)')
        
        if label is None:
            label =  self.inpath.split('/')[-1]
        plt.xlabel('Cluster Size (molecules)', fontsize=14)
        plt.ylabel('Bound fraction', fontsize=14)
        plt.xticks([0,1,2,3,4], labels=['0','10','$10^2$','$10^3$','$10^4$'])
        plt.yticks([0,0.5,1])
        #plt.title('FTC = 2000 uM (Inter + Intra)', fontsize=16)
        plt.title(label, fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
        
        
'''        
path = 'C:/Users/chatt/Desktop/pytest/NFsim/test_dataset/FTC_A5_B5_30uM_with_cb'      
bs = [5, 5]
mols = ['poly_A', 'poly_B']
bf_nfs = BoundFraction_NFsim(path, binding_sites=bs,molecules=mols)

print(bf_nfs) 
print()
bf_nfs.get_boundFraction(saveIt=True)
print()
  '''      
        


