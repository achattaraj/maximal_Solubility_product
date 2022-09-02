# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:26:28 2019

@author: Ani Chattaraj
"""

import re
import os, sys
from decimal import Decimal
import numpy as np
import pandas as pd
from csv import writer
from collections import defaultdict, OrderedDict
from time import time
from glob import glob
import matplotlib.pyplot as plt

#import multiprocessing as mp


def ProgressBar(jobName, progress, length=40):
    '''

    Parameters
    ----------
    jobName : String
        Name to display.
    progress : float
        Fraction of job done.
    length : integer, optional
        Length of the progress bar. The default is 40.

    Returns
    -------
    Displays a progress bar.

    '''
    completionIndex = round(progress*length)
    msg = "\r{} : [{}] {}%".format(jobName, "*"*completionIndex + "-"*(length-completionIndex), round(progress*100))
    if progress >= 1: msg += "\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    

def displayExecutionTime(func):
    """
    This decorator will calculate time needed to execute the "func" function 
    """
    def wrapper(*args, **kwrgs):
        t1 = time()
        func(*args, **kwrgs)
        t2 = time()
        delta = t2 - t1
        if delta < 60:
            print("Execution time : {:.4f} secs".format(delta))
        else:
            t_min, t_sec = int(delta/60), delta%60
            print(f"Execution time : {t_min} mins {t_sec} secs")
    return wrapper


class RunTimeStatistics:
    
    def __init__(self, txtfile, runs=None):
        '''

        Parameters
        ----------
        txtfile : String
            Location of simulation input file.
        runs : int, optional
            Number of runs. The default is None, which reads the number of runs from input file.

        Returns
        -------
        Collect and save execution times across multiple trials.

        '''
        self.simObj = ReadInputFile(txtfile)
        if runs == None:
            numRuns = self.simObj.getNumRuns()
            self.Runs = [int(i) for i in range(numRuns)]
        else:
            self.Runs = runs
        
    def __repr__(self):
        simfile = self.simObj.txtfile.split('/')[-1]
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}"
        return info
    
    @staticmethod
    def getRunTime(runfile):
        runTime, timeUnit = None, None
        with open(runfile, 'r') as tmpfile:
            line = tmpfile.readlines()[0]
            tps = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(tps) == 1:
                runTime = float(tps[0])
                timeUnit = "sec"
            elif len(tps) == 2:
                runTime = (float(tps[0])*60 + float(tps[-1]))/60
                timeUnit = "min"
            elif len(tps) == 3:
                runTime = (float(tps[0])*3600 + float(tps[1])*60 + float(tps[2]))/3600
                timeUnit = "hour"
            elif len(tps) == 4:
                runTime = (float(tps[0])*24*3600 + float(tps[1])*3600 + float(tps[2])*60 + float(tps[3]))/3600
                timeUnit = "hour"
        return runTime, timeUnit
     
    @displayExecutionTime 
    def getStat(self, fontsize=16):
        print("Getting runtime stats ...")
        seqNo, RT, units = [], [], []
        inpath = self.simObj.getInpath() + "/data"
        outpath = self.simObj.getOutpath("RunTime_stat")
        N_runs = len(self.Runs)
        for i, run in enumerate(self.Runs):
            try:
                rf = inpath + f"/Run{run}/RunningTime.txt"
                rt, unit = self.getRunTime(rf)
                seqNo.append(run)
                RT.append(rt)
                units.append(unit)
                ProgressBar("Progress", (i+1)/N_runs)
            except:
                pass
        #print(units)
        if len(set(units)) == 1: # when all units are of same type (sec/min/hour)
            if units[0] == "sec":
                plt.ylabel("Runtime (secs)", fontsize=fontsize)
            elif units[0] == "min":
                plt.ylabel("Runtime (mins)", fontsize=fontsize)
            elif units[0] == "hour":
                plt.ylabel("Runtime (hours)", fontsize=fontsize)
            plt.xlabel("Run Sequence", fontsize=fontsize)
            plt.scatter(seqNo, RT, color='b')
            with open(outpath + '/runTimeInfo.csv', 'w', newline='') as tmpf:
                wf = writer(tmpf)
                wf.writerows(zip(seqNo, RT))
            simName = self.simObj.txtfile.split('/')[-1].replace(".txt","")
            plt.show()
            plt.title(simName)
            plt.savefig(outpath + "/runTime.png", dpi=100)
            plt.close()
            print("Done plotting the stats ...!")
        else:
            print("Time units are different! Can't plot the runtime stats")
        

class ReadInputFile:
    
    def __init__(self, txtfile):
        self.txtfile = txtfile
    
    def readFile(self):
        with open(self.txtfile, 'r') as tmpfile:
            txtfile = [line for line in tmpfile.readlines()]
        return txtfile
    
    def getInpath(self):
        mypath = self.txtfile.split("/")[:-1]
        return "/".join(mypath)
    
    def getOutpath(self, statName):
        # statName : Name of the analysis; cluster_stat, e2e_stat etc
        inpath = self.getInpath()
        outpath = inpath + f"/pyStat/{statName}"
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        return outpath
    
    def getMolecules(self):
        lines = self.readFile()
        molNames, molCounts = [], []
        
        for index, line in enumerate(lines):
            if not re.search('[*]', line):
                if re.search('MOLECULE', line):
                    specs = line.split(':')[-1].split()
                    if len(specs)>2:
                        if int(specs[3]) != 0:  # molecules with non-zero count
                            molNames.append(specs[0].replace('"',''))
                            molCounts.append(int(specs[3]))
                            
        return molNames, molCounts
    
    @staticmethod
    def StringSearch(string, lines):
        value = None
        for line in lines:
            if re.search(string, line):
                value = line.split(':')[-1]
                break
        return value
    
    def getTimeStats(self):
        lines = self.readFile()
        t_tot = self.StringSearch("Total time", lines)
        ts = self.StringSearch("dt", lines)  # ts : time step for simulation
        t_data = self.StringSearch("dt_data", lines)
        t_image = self.StringSearch("dt_image", lines)
        return float(t_tot), float(ts), float(t_data), float(t_image)
    
    def getNumRuns(self):
        numRuns = 0
        lines = self.readFile()
        for line in lines:
            if re.search("\\b" + "Runs" + "\\b", line): # \\b : word boundary; avoids the line 'SimultaneousRuns: 1'
                numRuns = line.split(':')[-1]
                break
        return int(numRuns)



class ClusterAnalysis:

    def __init__(self, simfile, t_total=None, runs=None, withMonomer=True):
        '''

        Parameters
        ----------
        simfile : String
            Location of the simulation input file.
        t_total : Float, optional
            Last timepoint upto which analysis will be carried. The default is None, which reads the total time from input file.
        runs : Interger, optional
            Number of runs to include in the analysis. The default is None, which reads the number of Runs from input file.
        withMonomer : Boolean, optional
            Whether to include monomers (cluser size 1) while computing ACO. The default is True.

        Returns
        -------
        Computes average cluster occupancy of the cluster distributions across multiple runs over all the timepoints.

        '''
        
        self.simfileObj = ReadInputFile(simfile)

        tf, dt, dt_data, dt_image = self.simfileObj.getTimeStats()
        self.dt = dt_data
        if t_total == None:
            self.t_total = tf
        else:
            self.t_total = t_total
            
        self.withMonomer = withMonomer
        
        timepoints = np.arange(0, self.t_total+self.dt, self.dt)
        self.timePoints = timepoints
        
        numRuns = self.simfileObj.getNumRuns()
        if runs == None:
            self.runs = [run for run in range(numRuns)]
        elif type(runs) is int:
            self.runs = [i for i in range(runs)]
        elif type(runs) is list:
            self.runs = runs
        else:
            print("runs must be an integer or a list of runs")
        

    def __repr__(self):
        simfile = self.simfileObj.txtfile.split('/')[-1]
        info = f"\nClass : {self.__class__.__name__}\nSystem : {simfile}\nTotal_Time : {self.t_total} seconds\t\tnumRuns : {len(self.runs)}\n"
        return info

    @staticmethod
    def getSpeciesCount(df, species):
        try:
            # df : pandas dataframe
            col1, col2 = df[0], df[1]
            return [col2[i] for i in range(len(col1)) if col1[i]== species]
        except:
            pass

    def getClusterAverage(self, csvfile):
        # including the monomer
        acs, aco, foTM = 1, 1, {}
        clusterList = []
        df = pd.read_csv(csvfile, header=None)
        clusterCount = df[1][0]
        #col1, col2 = df[0], df[1]
        if clusterCount == 0:
            print('No cluster found')
        else:
            oligos = self.getSpeciesCount(df, "Size")
            if len(oligos) == 0:
                acs, aco = 1, 1
            else:
                monomerCount = clusterCount - len(oligos)
                clusterList.extend(oligos)
                clusterList.extend([1]*monomerCount)

                N, TM = len(clusterList), sum(clusterList)
                acs = TM/N # acs : average cluster size
                foTM = {clus: (clusterList.count(clus)*(clus/TM)) for clus in set(clusterList)} #foTM : fraction of total molecules
                aco = sum([cs*f for cs, f in foTM.items()])  # aco : average cluster occupancy

        return acs, aco, foTM


    def getOligoAverage(self, csvfile):
        # excluding the monomer
        acs, aco = 0, 0
        df = pd.read_csv(csvfile, header=None)
        oligos = self.getSpeciesCount(df, "Size")
        N, TMC = len(oligos), sum(oligos)  # TMC: total molecules in cluster
        try:
            acs = TMC/N
            foTM = {oligo: (oligos.count(oligo)*(oligo/TMC)) for oligo in set(oligos)}
            aco = sum([cs*f for cs, f in foTM.items()])
        except:
            pass # if there is no cluster > 1, acs = aco = 0
        return acs, aco


    @staticmethod
    def getTimeAverage(ndList, numRuns, numTimePoints):
        ndArray = np.array(ndList).reshape(numRuns, numTimePoints)
        timeAverage = np.mean(ndArray, axis=0, dtype=np.float64) #float64 gives more accurate mean
        return timeAverage

    
    @displayExecutionTime
    def getMeanTrajectory(self, SingleTraj=False):
        
        simObj = self.simfileObj
        inpath = simObj.getInpath()
        if not self.withMonomer:
            outpath = simObj.getOutpath('Cluster_stat_wom')
        else:
            outpath = simObj.getOutpath("Cluster_stat")

        nf = abs(Decimal(str(self.dt)).as_tuple().exponent) # Number of decimal points to format the clusterTime.csv file

        timepoints = self.timePoints
        numRuns = len(self.runs)
        numTimePoints = len(timepoints)
        
        if SingleTraj is False:
            csvfiles = [inpath + f"/data/Run{run}/Clusters_Time_{tp:.{nf}f}.csv" for run in self.runs for tp in timepoints]
            N = len(csvfiles)
            
            ave_clus_stat = [*map(lambda x: [self.getClusterAverage(x), ProgressBar('TimeCourse_calc', progress=(csvfiles.index(x)+1)/N)] , csvfiles)]
            ave_stat, _ = list(zip(*ave_clus_stat)) # acs_aco, None
            acs, aco, _ = list(zip(*ave_stat))
            
            print('Done')
    
            mean_acs =np.mean(np.array(acs).reshape((numRuns, numTimePoints)), axis=0)
            mean_aco =np.mean(np.array(aco).reshape((numRuns, numTimePoints)), axis=0)
            std_aco =np.std(np.array(aco).reshape((numRuns, numTimePoints)), axis=0)
    
            tp_ms = timepoints*1e3 # second to millisecond
    
            with open(outpath + "/Clustering_dynamics.csv","w", newline='') as outfile:
                wf = writer(outfile,delimiter=',')  # csv writer
                wf.writerow(['Time (ms)','ACS','ACO','std_ACO'])
                wf.writerows(zip(tp_ms, mean_acs, mean_aco, std_aco))
                outfile.close()
        else:
            print(f'Tracking one trajectory - Run{SingleTraj} ...')
            csvfiles = [inpath + f"/data/Run{SingleTraj}/Clusters_Time_{tp:.{nf}f}.csv" for tp in timepoints]
            N = len(csvfiles)
            ave_clus_stat = [*map(lambda x: [self.getClusterAverage(x), ProgressBar('Cluster_ave_calc', progress=(csvfiles.index(x)+1)/N)] , csvfiles)]
            
            ave_stat, _ = list(zip(*ave_clus_stat)) # acs_aco, None
            acs, aco, foTM = list(zip(*ave_stat))
            #print(foTM)
            #mean_acs =np.mean(np.array(acs))
            #mean_aco =np.mean(np.array(aco))
            
            tp_ms = timepoints*1e3 # second to millisecond
            #CD_timecourse = {k:v for k,v in zip(tp_ms, foTM)}
            #for k, v in CD_timecourse.items():
             #   print(k,v)
            
            with open(outpath + f"/Run{SingleTraj}_distribution_dynamics.csv","w", newline='') as of:
               wf = writer(of, delimiter=',')  # csv writer
               wf.writerow(['Time (ms)','foTM'])
               wf.writerows(zip(tp_ms, foTM))
               of.close()

    
            with open(outpath + f"/Run{SingleTraj}_clustering_dynamics.csv","w", newline='') as outfile:
                wf = writer(outfile,delimiter=',')  # csv writer
                wf.writerow(['Time (ms)','ACS','ACO'])
                wf.writerows(zip(tp_ms, acs, aco))
                outfile.close()


    @staticmethod
    def writeComposition(outpath, compo_dict, molecules):
        d = OrderedDict(sorted(compo_dict.items(), key = lambda x:x[0]))
        with open(outpath + "/Clusters_composition.txt","w") as tmpfile:
            tmpfile.write(f"Cluster Size \t {molecules} : frequency\n\n")
            for k, v in d.items():
                unique_comp = [list(x) for x in set(tuple(x) for x in v)]
                freq = [v.count(comp)/len(v) for comp in unique_comp]
                tmpfile.write(f"  {k}\t\t")
                for cmp, occur in zip(unique_comp, freq):
                    cmp = [str(s) for s in cmp]
                    tmpfile.write(",".join(cmp))
                    tmpfile.write(" : {:.2f}%\t".format(occur*100))
                tmpfile.write("\n\n")

    @staticmethod
    def writeDistribution(outpath, cluster_stat):
        TM_eff, N = sum(cluster_stat), len(cluster_stat)
        unique_clusters = sorted(set(cluster_stat))
        freq = [cluster_stat.count(clus)/N for clus in unique_clusters]
        foTM = [cluster_stat.count(clus)*(clus/TM_eff) for clus in unique_clusters]

        with open(outpath+"/SteadyState_distribution.csv", "w", newline='') as tmpfile:
            wf2 = writer(tmpfile)
            wf2.writerow(['Cluster size','frequency','foTM'])
            wf2.writerows(zip(unique_clusters, freq, foTM))
            tmpfile.close()

    @displayExecutionTime
    def getSteadyStateDistribution(self, SS_timePoints):
        #print("Getting steadystate cluster distribution ...")
        simObj = self.simfileObj
        inpath = simObj.getInpath()
        if self.withMonomer:
            outpath = simObj.getOutpath("Cluster_stat")
        else:
            outpath = simObj.getOutpath("Cluster_stat_wom")

        molecules, counts = simObj.getMolecules()
        nf = abs(Decimal(str(self.dt)).as_tuple().exponent) # Number of decimal points to format the clusterTime.csv file
        composition = defaultdict(list) # composition does not track the monomer identity
        #print("nf : ", nf)
        cluster_stat = []
        numRuns = len(self.runs)
        for j, run in enumerate(self.runs):
            for tp in SS_timePoints:
                df = pd.read_csv((inpath + "/data/Run{}/Clusters_Time_{:.{}f}.csv".format(run, tp, nf)), header=None)
                oligos = self.getSpeciesCount(df, "Size")
                mol_in_cluster = [self.getSpeciesCount(df, mol) for mol in molecules]
                for i, clus in enumerate(oligos):
                    comp = [mol_in_cluster[m][i] for m in range(len(mol_in_cluster))]
                    composition[clus].append(comp)

                if not self.withMonomer:
                    cluster_stat.extend(oligos)
                else:
                    monomerCount = df[1][0] - len(oligos)
                    monomers = [1]*monomerCount
                    cluster_stat.extend(monomers)
                    cluster_stat.extend(oligos)

            ProgressBar("Distribution_calc", (j+1)/numRuns)

        self.writeComposition(outpath, composition, molecules)
        self.writeDistribution(outpath, cluster_stat)

        with open(outpath + "/Sampling_stat.txt","w") as tmpfile:
            ss_tp1000 = [(t*1e3) for t in SS_timePoints]
            tmpfile.write("Number of runs : {}\n\n".format(len(self.runs)))
            tmpfile.write(f"Steady state time points (ms): {ss_tp1000}")
