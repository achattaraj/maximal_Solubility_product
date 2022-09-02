# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:33:23 2020

@author: Ani Chattaraj
"""

import numpy as np
from glob import glob
import sys, os
import re, json
from collections import defaultdict, OrderedDict
import csv
import time 

class NFSim_output_analyzer:
    def __init__(self, path):
        '''
        Parameters
        ----------
        path : File String
            DESCRIPTION: location of the source directory containing gdat files

        Returns
        -------
        None.

        '''
        self.path = path
    def __repr__(self):
        simfile = self.path.split('/')[-1]
        gfiles = glob(self.path + "\*.gdat")
        #sfiles = glob(self.path + "\*.species")
        info = f"\n***** // ***** \nClass : {self.__class__.__name__}\nSystem : {simfile}\nTotal Trajectories : {len(gfiles)}\n"
        return info

    #@displayExecutionTime
    def process_gdatfiles(self, m2cf_div=1, 
                          saveDist=False, printProgress=False):
        '''

        Computes Mean observable counts over multiple trajectories

        '''
        gfiles = glob(self.path + "\*.gdat")
        if len(gfiles) == 0:
            print('No gdat files found; quitting calculation ...')
            sys.exit()

        '''I use a test gdat file to extract the array dimension
            and name of the observables used in the model'''

        test_gf = gfiles[0]
        N_tp, N_obs = np.loadtxt(test_gf).shape # number of timepoints and observables

        with open(test_gf,'r') as tmpf:
            obs_names = tmpf.readline().split()[1:]
            obs_names = '\t'.join(obs_names)


        '''The temporary matrix would store the data from multiple trajectories
            and perform the average'''

        tmp_matrix = np.empty(shape=(len(gfiles),N_tp, N_obs), dtype=float)
        N_gf = len(gfiles)

        for i, gf in enumerate(gfiles):
            data = np.loadtxt(gf)
            data[:,1:] = data[:,1:]/m2cf_div # change to conc units except the time row
            tmp_matrix[i] = data
            if printProgress:
                self.ProgressBar('Processing gdat_files', (i+1)/N_gf)

        mean_obs = np.mean(tmp_matrix, axis=0)
        std_obs = np.std(tmp_matrix, axis=0)
        outpath = self.getOutpath()
        np.savetxt(outpath + "\Mean_Observable_Counts.txt", mean_obs, header=obs_names, fmt='%.4e')
        np.savetxt(outpath + "\Stdev_Observable_Counts.txt", std_obs, header=obs_names, fmt='%.6e')
        
        if saveDist:
            # slice: run, timepoints, observables
            stat_fa4 = tmp_matrix[:,[5,10,15,20],[2]].flatten() # four timepoints: 5,10,15,20 ms
            stat_fb4 = tmp_matrix[:,[5,10,15,20],[4]].flatten()
            stat_ = np.array([stat_fa4, stat_fb4])
            np.savetxt(outpath + "\multiTraj_dist.txt", stat_.T,header='free A4\tfree B4',fmt='%.4e')

    #@displayExecutionTime
    def process_speciesfiles(self, molecules=[], saveTraj=False,
                             calcRatio=False, printProgress=False):

        '''

        molecules = List of molecules used in the model
        saveTraj = save clusters from single trajectories
        calcRatio = calculates composition ratio of clusters (A4:B4 in clusters)
        -------

        Computes distribution of molecular clusters and their compositions

        '''
        sfiles = glob(self.path + "/*.species")
        flatten_ = lambda myList: [item for sublist in myList for item in sublist]
        cs_stat, comp_stat = defaultdict(list), defaultdict(list)
        ratio_stat = defaultdict(list)
        N_sp = len(sfiles)
        TrajStat = {} # saving clusters from single runs

        for i, sf in enumerate(sfiles):
            cs, comp = self.collect_clusters(sf, molecules)
            if saveTraj:
                seqNum = sf.split('/')[-1].split('\\')[-1].replace(".species","")
                TrajStat[seqNum] = cs
            
            for size, count in cs.items():
                cs_stat[size].append(count)
            if calcRatio:    
                for size, composition in comp.items():
                    comp_stat[size].append(composition)
                    for cmp in composition:
                        if cmp[0] + cmp[1] > 1: # avoiding monomers
                            ratio_stat[size].append(cmp[0]/cmp[1])
            else:
                for size, composition in comp.items():
                    comp_stat[size].append(composition)
                
            if printProgress:    
                self.ProgressBar('Processing species_files', (i+1)/N_sp)

        cs_stat = {k: sum(v) for k, v in cs_stat.items()}
        comp_stat = {k: flatten_(v) for k, v in comp_stat.items()}

        outpath = self.getOutpath()
        #print(TrajStat)
        if saveTraj:
            with open(outpath + "/SingleTraj_clusters.json", "w") as tf:
                json.dump(TrajStat, tf, sort_keys=True, indent="")
        if calcRatio:
            with open(outpath + "/Cluster_composition_ratio.json", "w") as tf:
                json.dump(ratio_stat, tf, sort_keys=True, indent="")

        self.writeComposition(outpath, comp_stat, molecules)
        self.writeDistribution(outpath, cs_stat)

    def getOutpath(self):
        outpath = self.path + "/pyStat"
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        return outpath

    @staticmethod
    def collect_clusters(speciesFile, molecules):
        '''
        Parameters
        ----------
        speciesFile : File String
            DESCRIPTION: Speciesfile containing all the molecular species
        molecules : List of String
            DESCRIPTION: List of molecules used in the model

        Returns
        -------
        A pair of defaultdicts; one with cluster size distribution
        and another with the corresponding compositions of the clusters

        '''
        try:
            with open(speciesFile, 'r') as tf:
                currentFrame = tf.readlines()[2:] # to avoid first two warning lines
        except:
            print("File missing: ", speciesFile)
            sys.exit()
        else:
            clus_stat = defaultdict(list)
            comp_stat = defaultdict(list)
            for line in currentFrame:
                if not (line == '\n' or re.search('Time', line) or re.search('Sink', line) or re.search('Source', line)):
                    cluster, count = line.split()
                    comp = tuple([cluster.count(mol) for mol in molecules])
                    cs = len(cluster.split('.')) + 1 # monomer does not have bonds (.)
                    
                    clus_stat[cs].append(int(count))
                    comp_stat[cs].append(comp)
        clus_stat = {k: sum(v) for k,v in clus_stat.items()}
        return clus_stat, comp_stat


    @staticmethod
    def writeDistribution(outpath, cluster_stat):
        '''
        Parameters
        ----------
        outpath : File String
            DESCRIPTION: Location of the output files
        cluster_stat : Defaultdict
            DESCRIPTION: Dictionary with {keys, values} = {cluster size, occurence}

        Returns
        -------
        None.
        '''
        cluster_stat = OrderedDict(sorted(cluster_stat.items(), key = lambda x:x[0]))
        TC = sum(cluster_stat.values()) # total counts
        TM = sum([k*v for k,v in cluster_stat.items()])  # total molecules
        #print('TM = ', TM, ' TC = ',  TC)
        foTM = {cs: count*(cs/TM) for cs,count in cluster_stat.items()} # fraction of total molecules
        occurence = {cs: count/TC for cs, count in cluster_stat.items()}

        with open(outpath + "/Cluster_frequency.csv","w", newline='') as tmpfile:
            wf = csv.writer(tmpfile)
            wf.writerow(['Cluster size','counts'])
            wf.writerows(zip(cluster_stat.keys(),cluster_stat.values()))

        with open(outpath+"/SteadyState_distribution.csv", "w", newline='') as tmpfile:
            wf2 = csv.writer(tmpfile)
            wf2.writerow(['Cluster size','frequency','foTM'])
            wf2.writerows(zip(cluster_stat.keys(), occurence.values(), foTM.values()))

    @staticmethod
    def writeComposition(outpath, compo_dict, molecules):
        '''
        Parameters
        ----------
        outpath : File String
            DESCRIPTION: Location of the output files
        compo_dict : Defaultdict
            DESCRIPTION: Dictionary with {keys, values} = {cluster size, compositions}
        molecules : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        d = OrderedDict(sorted(compo_dict.items(), key = lambda x:x[0]))
        with open(outpath + "/Clusters_composition.txt","w") as tmpfile:
            tmpfile.write(f"Cluster Size \t {molecules} : frequency\n\n")

            for k, v in d.items():
                unique_comp = set(v)
                freq = [v.count(uc)/len(v) for uc in unique_comp]
                tmpfile.write(f"  {k}\t\t")
                for cmp, occur in zip(unique_comp, freq):
                    cmp = [str(s) for s in cmp]
                    tmpfile.write(",".join(cmp))
                    tmpfile.write(" : {:.2f}%\t".format(occur*100))

                tmpfile.write("\n\n")

    @staticmethod
    def ProgressBar(jobName, progress, length=40):

        '''
        Parameters
        ----------
        jobName : string
            Name of the job given by user.

        progress : float
            progress of the job to be printed as percentage.

        length : interger
            prints the length of the progressbar. The default is 40.

        Returns
        -------
        None.
        '''
        completionIndex = round(progress*length)
        msg = "\r{} : [{}] {}%".format(jobName, "*"*completionIndex + "-"*(length-completionIndex), round(progress*100))
        if progress >= 1: msg += "\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()

    @staticmethod
    def displayExecutionTime(func):
        """
        This decorator (function) will calculate the time needed to execute a task
        """
        def wrapper(*args, **kwrgs):
            t1 = time.time()
            func(*args, **kwrgs)
            t2 = time.time()
            delta = t2 - t1
            if delta < 60:
                print("Execution time : {:.4f} secs".format(delta))
            else:
                t_min, t_sec = int(delta/60), delta%60
                print(f"Execution time : {t_min} mins {t_sec} secs")

        return wrapper
                                       
'''
if __name__ == '__main__':
    
    paths = glob('C:/Users/chatt/Desktop/pytest/NFsim/test_dataset/FTC_A5_B5_30uM_with_cb')
    N = len(paths)
    
    #c2mf = [200.0, 100.0, 66.66, 50.0, 40.0, 33.33, 28.57, 25.0, 22.22, 20.0]
    for i, path in enumerate(paths):
        molecules = ['poly_A', 'poly_B']
        #molecules = ['poly_SH3', 'poly_PRM']
        #molecules = ['Nephrin', 'Nck', 'NWASP']
        
        nfs_obj = NFSim_output_analyzer(path)
        print(nfs_obj)
        try:
            nfs_obj.process_gdatfiles(m2cf_div=10,saveDist=False,printProgress=True)
            nfs_obj.process_speciesfiles(molecules,calcRatio=False,printProgress=True)
        except:
            print('Incomplete: ', path)
            
        #nfs_obj.ProgressBar('progress', (i+1)/N, length=40)
'''    
         