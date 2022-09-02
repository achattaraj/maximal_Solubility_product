# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:25:41 2020

@author: Ani Chattaraj
"""

import numpy as np
import os, re, sys
from time import time
from csv import writer, reader
#from AllBondPlot import PlotAllBond
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob


class MoleculeCounter:
    
    def __init__ (self, inpath, numRuns=[]):
        self.inpath = inpath
        if type(numRuns) == list:
            self.runs = numRuns
        if type(numRuns) == int:
            self.runs = [_ for _ in range(numRuns)]
        if len(numRuns) == 0:
            self.runs = 0
    
    
    def getOutpath(self):
        outpath = self.inpath + "/pyStat/Count_Stat"
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        return outpath
    
    @staticmethod
    def getNumRuns(lines):
        numRuns = 0
        for line in lines:
            if re.search("\\b" + "Runs" + "\\b", line): # \\b : word boundary; avoids the line 'SimultaneousRuns: 1'
                numRuns = line.split(':')[-1]
                break
        return int(numRuns)

    
    def getBoxSize(self):
        txtfile = self.inpath.replace('\\','/').split('/')[-1].replace('_FOLDER','.txt')
        
        with open(self.inpath + "/" + txtfile, 'r') as tf:
            for line in tf.readlines():
                if re.search('L_x', line):
                    Lx = float(line.split(':')[-1].strip())
                if re.search('L_y', line):
                    Ly = float(line.split(':')[-1].strip())
                if re.search('L_z_in', line):
                    Lz = float(line.split(':')[-1].strip())
                
                if re.search("\\b" + "Runs" + "\\b", line): # \\b : word boundary; avoids the line 'SimultaneousRuns: 1'
                    numRuns = line.split(':')[-1]
                    break
            
            
        return (Lx*1e3,Ly*1e3,Lz*1e3), int(numRuns)  # in nm
    
    @staticmethod
    def getMoleculeCount(file, mols):
        mol_stat = []
        df = pd.read_csv(file, header=0)
        for mol in mols:
            mol_stat.append(list(df[mol]))
        
        return np.asarray(mol_stat)
    
    def getTimePoints(self):
        file = None
        for run in self.runs:
            testfile = self.inpath + f"/data/Run{run}/FullBondData.csv"
            #print(testfile)
            if os.path.isfile(testfile):
                file = testfile
                break
            else:
                pass
        else:
            print('No run is complete yet')
        
        if file is None:
            return None
        else:
            df = pd.read_csv(file)
            return df['Time']
            
        
    def getMoleculeStat(self, molecules):
        print('Getting molecular counts...')
        molCount = []
        incmp_runs = []
        
        #tp = self.getTimePoints()
        tp = []
        
        (lx,ly,lz), numRun = self.getBoxSize()
        
        sysVol = lx*ly*lz
        
        if self.runs == 0:
            self.runs = [_ for _ in range(numRun)]
        
        for i, run in enumerate(self.runs):
            file = self.inpath + "/data/Run{}/FullCountData.csv".format(run)
            
            df = pd.read_csv(file, header=0)
            tp = df['Time']
            mols = self.getMoleculeCount(file, molecules)
            #print(mols)
            molCount.append(mols)
            
           
            
        if len(incmp_runs) > 0:    
            print('Incomplete Runs: ', incmp_runs)
    
        factor = 6.023 * 1e-7  # converts uM to molecules/nm3
        molConc_arr = np.asarray(molCount)/(factor*sysVol)
        
        
        molCount_arr = np.asarray(molCount)
        
        _, numMols, _ = molConc_arr.shape # shape : numRuns, numMolecules, numTimepoints
        
        mean_molC = np.mean(molConc_arr, axis=0) # axis = 0 gives average molecular counts over multiple trajectories
        std_molC = np.std(molConc_arr, axis=0)
        
        mean_mols = np.mean(molCount_arr, axis=0) # axis = 0 gives average molecular counts over multiple trajectories
        std_mols = np.std(molCount_arr, axis=0)
        
        df = pd.DataFrame({"Time":[_ for _ in tp]})
        df2 = pd.DataFrame({"Time":[_ for _ in tp]})
        
        
        for i in range(numMols):
            df[molecules[i] + ' (mean uM)'] = [_ for _ in mean_molC[i]]
            df[molecules[i] + ' (stdev uM)'] = [_ for _ in std_molC[i]]
            
            df2[molecules[i] + ' (mean count)'] = [_ for _ in mean_mols[i]]
            df2[molecules[i] + ' (stdev)'] = [_ for _ in std_mols[i]]
            
            plt.plot(tp*1e3, mean_molC[i], label=molecules[i])
            plt.legend(fontsize=16)
            
        df.to_csv(self.getOutpath() + "/Molecular_Concentration.csv", sep=',')
        df2.to_csv(self.getOutpath() + "/Molecular_Counts.csv", sep=',')
        print('Wrote data! Filepath: ' + self.getOutpath() + "/Molecular_Concentration.csv")
        plt.xlabel('Time (ms)', fontsize=16)
        plt.ylabel('Free concentration ($\mu$M)', fontsize=16)
        plt.show()
 
        
''' 
path = 'C:/Users/chatt/Desktop/pytest/springsalad/test_dataset/A5_B5_flex_3nm_2nm_count_40_SIM_FOLDER' 

mols = ['FREE poly_A', 'FREE poly_B']

mc = MoleculeCounter(path)
mc.getMoleculeStat(mols)
'''
 
      
            