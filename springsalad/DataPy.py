# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:27:54 2019

@author: Ani Chattaraj
"""
import re, json 
import os, sys
from decimal import Decimal
import numpy as np
import pandas as pd
from csv import writer
from collections import defaultdict, OrderedDict
from time import time
from glob import glob
#from DataPyPlot import DataPyPlot
import matplotlib.pyplot as plt
import multiprocessing as mp

def ProgressBar(jobName, progress, length=40):
    completionIndex = round(progress*length)
    msg = "\r{} : [{}] {}%".format(jobName, "*"*completionIndex + "-"*(length-completionIndex), round(progress*100))
    if progress >= 1: msg += "\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

def displayExecutionTime(func):
    """
    This decorator will calculate time needed to execute a function 
    """
    def wrapper(*args, **kwrgs):
        t1 = time()
        func(*args, **kwrgs)
        t2 = time()
        delta = t2 - t1
        if delta < 60:
            print("Execution time : {:.4f} secs".format(delta))
        else:
            t_min, t_sec = int(delta/60), round(delta%60)
            print(f"Execution time : {t_min} mins {t_sec} secs")
    return wrapper

def getSteadyStateIndicies(timeSeries, ss_timepoints, rel_tol=1e-8):
    # both are lists of float numbers; so cant' apply equally directly
    ss_index = []
    for i, elem in enumerate(timeSeries):
        if any([np.isclose(elem, t, rtol=rel_tol) for t in ss_timepoints]):
            # returns true if finds atleast one match
            ss_index.append(i)
    return ss_index

def getCompleteRunList(txtfile):
    rif = ReadInputFile(txtfile)
    path = rif.getInpath() + "/pyStat"
    with open(path + "/LCR.txt", 'r') as tmpfile:
        lcr = tmpfile.readlines()[0]
        LCR = lcr.split(',')[:-1]
        LCR = [int(r) for r in LCR]
    return LCR

    

class ReadInputFile:
    
    def __init__(self, txtfile):
        self.txtfile = txtfile
    
    def readFile(self):
        with open(self.txtfile, 'r') as tmpfile:
            txtfile = [line for line in tmpfile.readlines()]
        return txtfile
    
    def getInpath(self):
        mypath = self.txtfile.replace('\\','/').split("/")[:-1]
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
    
    def getReactiveSites(self):
        reactive_sites = []
        lines = self.readFile()
        n1, n2 = 0, 0
        for index, line in enumerate(lines):
            if re.search("BINDING REACTIONS", line):
                n1 = index
            elif re.search("MOLECULE COUNTERS", line):
                n2 = index
        binding_rxns = lines[n1+1 : n2]
        for line in binding_rxns:
            if line != '\n':
                lhs, rhs = [elem for elem in line.split('+')]
                lhs = lhs.split(':')
                rhs = rhs.split(':')
                reactive_sites.append(lhs[-2])
                reactive_sites.append(rhs[1])
        
        reactive_sites = [site.replace("'","").strip() for site in reactive_sites]
        reactive_sites = [site for site in set(reactive_sites)]
        
        return reactive_sites
    

class ClusterAnalysis:
    
    def __init__(self, simfile, t_total=None, runs=None, withMonomer=True):
        
        """
        simfile: string 
        t_total: float;
        """
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
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}\nTotal_Time : {self.t_total} seconds\t\tnumRuns : {len(self.runs)}"
        return info
    
    @staticmethod        
    def getSpeciesCount(df, species):
        # df : pandas dataframe
        col1, col2 = df[0], df[1]
        return [col2[i] for i in range(len(col1)) if col1[i]== species]
    
    def getClusterAverage(self, csvfile):
        # including the monomer
        acs, aco, foTM = 0, 0, 0
        clusterList = []
        df = pd.read_csv(csvfile, header=None)
        clusterCount = df[1][0]
        #col1, col2 = df[0], df[1]
        if clusterCount == 0:
            pass
            #print('\nNo cluster found')
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

        #print(csvfile)
        #print('foTM', foTM)
        return acs, aco
    
    def getMonomerCounts(self, csvfile, mols=[], counts=[]):
        df = pd.read_csv(csvfile, header=None)
        mols_in_clus = [sum(self.getSpeciesCount(df, mol)) for mol in mols]
        monomers = np.array(counts) - np.array(mols_in_clus)
        return list(monomers)
    
    
    def getOligoAverage(self, csvfile):
        # excluding the monomer
        acs, aco = 1, 1
        oligoList = []
        df = pd.read_csv(csvfile, header=None)
        oligos = self.getSpeciesCount(df, "Size")
        N, TMC = len(oligos), sum(oligos)  # TMC: total molecules in cluster
        try:
            acs = TMC/N
            foTM = {oligo: (oligos.count(oligo)*(oligo/TMC)) for oligo in set(oligos)}
            aco = sum([cs*f for cs, f in foTM.items()])
        except:
            pass # if there is no cluster > 1, acs = aco = 1
        return acs, aco
    
    ''' 
    @staticmethod
    def getTimeAverage(ndList, numRuns, numTimePoints):
        ndArray = np.array(ndList).reshape(numRuns, numTimePoints)
        timeAverage = np.mean(ndArray, axis=0, dtype=np.float64) #float64 gives more accurate mean
        return timeAverage
    
    @displayExecutionTime
    def getMeanTrajectory(self):
        print("Getting Clustering Timecourse ...")
        simObj = self.simfileObj
        inpath = simObj.getInpath()
        if not self.withMonomer:
            outpath = simObj.getOutpath('Cluster_stat_wom')
        else:
            outpath = simObj.getOutpath("Cluster_stat")
        
        nf = abs(Decimal(str(self.dt)).as_tuple().exponent) # Number of decimal points to format the clusterTime.csv file
        acs_stat, aco_stat = [], []
        timepoints = np.arange(0, self.t_total+self.dt, self.dt)
        numRuns = len(self.runs)
        for i, run in enumerate(self.runs):
            for tp in timepoints:
                df = pd.read_csv((inpath + "/data/Run{}/Clusters_Time_{:.{}f}.csv".format(run, tp, nf)), header=None)
                if self.withMonomer:
                    acs, aco = self.getClusterAverage(df, tp, run) # need to change this func
                else:
                    acs, aco = self.getOligoAverage(df)
                
                acs_stat.append(acs)
                aco_stat.append(aco)
            ProgressBar("Progress", (i+1)/numRuns)
        
        mean_acs = self.getTimeAverage(acs_stat, numRuns, len(timepoints))
        mean_aco = self.getTimeAverage(aco_stat, numRuns, len(timepoints))
        
        tp_ms = timepoints*1e3 # second to millisecond
        
        with open(outpath+"/Clustering_dynamics.csv","w", newline='') as outfile:
            wf = writer(outfile,delimiter=',')  # csv writer
            wf.writerow(['Time (ms)','ACS','ACO'])
            wf.writerows(zip(tp_ms,mean_acs,mean_aco))
            outfile.close()
    '''
    
    @displayExecutionTime
    def getMeanTrajectory_modified(self, saveClusDist=False, process='SEQ', saveMultiTraj=False):
        # process : 'seq' (sequential) or 'par' (parallel)
        # saveClusdist = saves timecourse of cluster distribution as a function of time
        print("Getting Clustering Timecourse by modified method ...")
        simObj = self.simfileObj
        molecules, counts = simObj.getMolecules()
        inpath = simObj.getInpath()
        if not self.withMonomer:
            outpath = simObj.getOutpath('Cluster_stat_wom')
        else:
            outpath = simObj.getOutpath("Cluster_stat")

        nf = abs(Decimal(str(self.dt)).as_tuple().exponent) # Number of decimal points to format the clusterTime.csv file

        timepoints = self.timePoints
        numRuns = len(self.runs)
        numTimePoints = len(timepoints)
        csvfiles = [inpath + f"/data/Run{run}/Clusters_Time_{tp:.{nf}f}.csv" for run in self.runs for tp in timepoints]
        N = len(csvfiles)

        if process.upper() == 'SEQ':
            print('Sequential processing...')
            #ave_clus_stat = list(map(self.getClusterAverage, csvfiles))
            if not self.withMonomer:
                ave_clus_stat = [*map(lambda x: [self.getOligoAverage(x), ProgressBar('Oligo_ave_calc', progress=(csvfiles.index(x)+1)/N)] , csvfiles)]
            else:
                ave_clus_stat = [*map(lambda x: [self.getClusterAverage(x), self.getMonomerCounts(x,molecules,counts), ProgressBar('Cluster_ave_calc', progress=(csvfiles.index(x)+1)/N)] , csvfiles)]
                
            ave_stat, monomers, _ = list(zip(*ave_clus_stat)) # acs_aco, None
            acs, aco = list(zip(*ave_stat))
            

        elif process.upper() == 'PAR':
            print('Parallel processing...')
            pool = mp.Pool(4)
            #ave_clus_stat = pool.starmap(lambda x: [self.getClusterAverage(x), ProgressBar('Cluster_ave_calc', progress=(csvfiles.index(x)+1)/N)] , csvfiles)
            if not self.withMonomer:
                ave_clus_stat = pool.map(self.getOligoAverage, csvfiles)
            else:
                ave_clus_stat = pool.map(self.getClusterAverage, csvfiles)
            
            acs, aco = list(zip(*ave_clus_stat))

        print('Done')

        
        #print(foTM)
        mean_acs =np.mean(np.array(acs).reshape((numRuns, numTimePoints)), axis=0)
        mean_aco =np.mean(np.array(aco).reshape((numRuns, numTimePoints)), axis=0)
        stdev_aco =np.std(np.array(aco).reshape((numRuns, numTimePoints)), axis=0)
        #mean_monomers = np.mean(np.array(monomers).reshape((numRuns, numTimePoints)), axis=0)
        
        
        mono_traj = np.split(np.array(monomers), numRuns)
        m_traj = np.mean(mono_traj, axis=0)
        #print(m_traj)
        tp_ms = timepoints*1e3 # second to millisecond
        
        if saveMultiTraj:
            aco_trajs = np.array(aco).reshape((numRuns, numTimePoints)).T
            
            run_counts = [f'Run_{run}' for run in range(numRuns)]
            header = '\t'.join(run_counts)
            
            with open(outpath + "/MultiTraj_stat.txt", "w") as tmf00:
                np.savetxt(tmf00, aco_trajs, fmt='%.3e', header=header)
        
        with open(outpath + "/Monomer_counts.txt", "w") as tmf01:
            head_string =  '\t'.join(molecules)
            #tp_tmp = [[t] for t in tp_ms]
            #np.insert(m_traj, [:0], tp_ms, axis=1)
            np.savetxt(tmf01, m_traj, fmt='%.5e', header=head_string)
            #print(m_traj)

        with open(outpath + "/Clustering_dynamics.csv","w", newline='') as outfile:
            wf = writer(outfile,delimiter=',')  # csv writer
            wf.writerow(['Time (ms)','ACS','ACO','Stdev_ACO'])
            wf.writerows(zip(tp_ms, mean_acs, mean_aco, stdev_aco))
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
        print("Getting steadystate cluster distribution ...")
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
        clusTraj = {}
        for j, run in enumerate(self.runs):
            clusDict = {}
            for tp in SS_timePoints:
                try:
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
                        clusDict['1'] = str(monomerCount)
                        tmp = {str(c):str(oligos.count(c)) for c in oligos}
                        clusDict.update(tmp)
                        clusTraj[f'Run_{run}'] = clusDict
                        
                        monomers = [1]*monomerCount
                        cluster_stat.extend(monomers)
                        cluster_stat.extend(oligos)
                except:
                    print(f'Run = {run}, timepoint = {tp}, missing')
                    pass
            
            ProgressBar("Progress", (j+1)/numRuns)
        #print(clusTraj)
        self.writeComposition(outpath, composition, molecules)
        self.writeDistribution(outpath, cluster_stat)
        
        with open(outpath + "/Sampling_stat.txt", "w") as tmpfile:
            ss_tp1000 = [(t*1e3) for t in SS_timePoints]
            tmpfile.write("Number of runs : {}\n\n".format(len(self.runs)))
            tmpfile.write(f"Steady state time points (ms): {ss_tp1000}")
       
        with open(outpath + f"/TrajClusterDist.json", "w") as tmf:
            json.dump(clusTraj, tmf,  indent="")
            
        
class InterSiteDistance:
    
    def __init__(self, simfile, site1_name=None, site2_name=None):
        self.simfileObj = ReadInputFile(simfile)
        if (site1_name is None) or (site2_name is None):
            print("Please provide both the site names")
        self.siteName1 = site1_name
        self.siteName2 = site2_name
        tf, dt, dt_data, dt_image = self.simfileObj.getTimeStats()
        timeSeries = np.arange(0, tf+dt_image, dt_image)
        self.timeSeries = timeSeries
        #print(self.timeSeries)
        self.N_frame = round(tf/dt_image)  # number of frames in vierwervile
        
        
    def __repr__(self):
        simfile = self.simfileObj.txtfile.split('/')[-1]
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}\nSite1 : {self.siteName1}\t\tSite2 : {self.siteName2}"
        return info
    
    @staticmethod
    def findFile(inpath, numRuns, fileName):
        siteIDfile = None
        for run in range(numRuns):
            fname = inpath + f"/Run{run}/{fileName}.csv"
            if os.path.isfile(fname):
                siteIDfile = fname
                break
            else:
                pass
        if siteIDfile is None:
            print(f"Could not locate SiteIDs.csv across {numRuns} runs")
        else:
            return siteIDfile
        
    
    def getSiteIDs(self):
        id1, id2 = None, None 
        inpath = self.simfileObj.getInpath() + "/data"
        numRuns = self.simfileObj.getNumRuns()
        idFile = self.findFile(inpath, numRuns, "SiteIDs")
        #print('idfile: ', idFile)
        with open(idFile, 'r') as infile:
            '''
            r'\b{word}\b' would find the exact 'word' in lines,
            avoiding 'words' or 'word2' or 'word_nv' etc
            '''
            for line in infile:
                #if re.search(self.siteName1, line, re.IGNORECASE):
                if re.search(r'\b{}\b'.format(self.siteName1), line):
                    #print(f'found {self.siteName1} in {line}')
                    id1 = line.split(',')[0]
                    break
            for line in infile:
                #if re.search(self.siteName2, line, re.IGNORECASE):
                if re.search(r'\b{}\b'.format(self.siteName2), line):
                    #print(f'found {self.siteName2} in {line}')
                    id2 = line.split(',')[0]
                    break
        
        print(f'id1: {id1} \t id2: {id2}')
        
        if id1 is None:
            print(f"Invalid site name! Could not find {self.siteName1}")
        elif id2 is None:
            print(f"Invalid site name! Could not find {self.siteName2}")
        else:
            return id1, id2
        
    
    @staticmethod
    def getSiteLocation(viewerfile, siteID):
        pos = []
        vfile = open(viewerfile, 'r')
        for line in vfile.readlines():
            if (re.search("ID",line) and re.search(siteID, line)):
                coors = line.split()
                x,y,z = float(coors[-1]), float(coors[-2]), float(coors[-3])
                pos.append((x,y,z))
        vfile.close()
        return pos
    
    @staticmethod
    def getDistance(pos1, pos2):
        dist = []
        for p1, p2 in zip(pos1, pos2):
            dx,dy,dz = p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]
            d = np.sqrt(dx*dx + dy*dy +dz*dz)
            dist.append(d)
        return dist
    @displayExecutionTime
    def calculate_distance(self, ss_timepoints=[], writeIt=False):
        print(f"Calculating the distance between {self.siteName1} and {self.siteName2} ...")
        dist_stat = []
        id1, id2 = self.getSiteIDs()
        tf, dt, dt_data, dt_image = self.simfileObj.getTimeStats()
        """
        reverse the dt_image string and then find the decimal; 
        0.02->20.0, count = 2
        0.005 -> 500.0, count = 3
        """
        #decimal_count = "{}".format(dt_image)[::-1].find('.') # for formatting timepoints while writing data
        
        inpath = self.simfileObj.getInpath()
        outpath = self.simfileObj.getOutpath("ISD_stat")
        completeTrajCount = 0 # checks if any timepoint gets skipped within a viewerfile
        vfiles =  glob(inpath + "/viewer_files/*.txt")
        IVF = [] # Incomplete Viewer File
        N_traj = len(vfiles)
        
        for index, vfile in enumerate(vfiles):
            pos1 = self.getSiteLocation(vfile, id1)
            pos2 = self.getSiteLocation(vfile, id2)
            dist = self.getDistance(pos1, pos2)
            if len(dist) == self.N_frame + 1:
                dist_stat.append(dist)
                completeTrajCount += 1
            else:
                traj_name = vfile.split('/')[-1]
                IVF.append(traj_name)
            ProgressBar("Progress", (index+1)/N_traj)
        
        if len(IVF) > 0:
            print("incomplete trajectories: ", len(IVF))
            
        dist_arr = np.array(dist_stat)
        mean_dist = np.mean(dist_arr, axis=0, dtype=np.float64)
        ss_index = getSteadyStateIndicies(self.timeSeries, ss_timepoints=ss_timepoints)
        SS_dist = dist_arr[:,ss_index]
        
        if writeIt:
            t1000 = [t*1e3 for t in self.timeSeries]
            ss_tp1000 = [t*1e3 for t  in ss_timepoints]
            with open(outpath + f"/{self.siteName1}_{self.siteName2}_ISD_dynamics.csv","w", newline='') as outfile1, open(outpath + f"/{self.siteName1}_{self.siteName2}_ISD_distribution.txt","w") as outfile2:
                wf1 = writer(outfile1,delimiter=',')  # csv writer
                wf1.writerow(['Time(ms)','ISD(nm)'])
                wf1.writerows(zip(t1000, mean_dist))
                
                np.savetxt(outfile2, SS_dist, fmt="%.4e", delimiter=',')
            with open(outpath + "/Sampling_stat.txt","w") as of:
                print("Complete trajectories : ", completeTrajCount)
                of.write(f"Steady state time points (ms) : {ss_tp1000}\n\n")
                of.write(f"Number of complete trajectories : {completeTrajCount}\n\n")
                if len(IVF) > 0:
                    #print("Incomplete trajectories:")
                    of.write("Incomplete trajectories:\n\n")
                    for traj_name in IVF:
                        name = traj_name.split('/')[-1]
                        #print (name)
                        of.write(f"{name}\n\n")
   
            print("\nWriting data is complete\n")

class SingleMoleculeTracking:
    
    def __init__(self, simfile, moleculeName, RadGy_calc=True, MSD_calc=True, coordinate=None):
        self.simfileObj = ReadInputFile(simfile)
        tf, dt, dt_data, dt_image = self.simfileObj.getTimeStats()
        timeSeries = np.arange(0, tf+dt_image, dt_image)
        self.timeSeries = timeSeries
        #self.timeSeries = [int(t*1e3) for t in timeSeries]
        #print(self.timeSeries)
        self.N_frame = round(tf/dt_image)  # number of frames in vierwervile
        self.molName = moleculeName
        self.radGy_calc = RadGy_calc
        self.msd_calc = MSD_calc
        self.coor = coordinate
        print(f"tf = {tf}, dt_image = {dt_image}, N_frame = {self.N_frame}")
        
    def __repr__(self):
        simfile = self.simfileObj.txtfile.split('/')[-1]
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}\nMolecule: {self.molName}\nRadGy: {self.radGy_calc}\t\tMSD_calc: {self.msd_calc}\t dimension: {self.coor}"
        return info
    
    @staticmethod
    def getMolecularSiteIDs(inpath, numRuns, molName):
        idFile = InterSiteDistance.findFile(inpath, numRuns, "SiteIDs")
        ids = []
        with open(idFile, 'r') as tmpfile:
            for line in tmpfile.readlines():
                if line.split(',')[-1].split()[0] == molName:
                    sid = line.split(',')[0]
                    ids.append(sid)
        return ids
        
    @staticmethod
    def getCoordinates(viewerfile, siteIdList):
        frame_indices = []
        Rg = []
        mcList = []  # coordinates of molecular centroids
        with open(viewerfile, 'r') as tmpfile:
            lines = tmpfile.readlines()
            for i, line in enumerate(lines):
                if re.search("SCENE", line):
                    frame_indices.append(i)
            frame_indices.append(len(lines))
            j = 0
            while(j < len(frame_indices) - 1):
                cur_frame = lines[frame_indices[j]:frame_indices[j+1]]
                j += 1
                coors = []
                for line in cur_frame:
                    if re.search("ID", line):
                        line = line.split()
                        if line[1] in siteIdList:
                            coors.append((float(line[-1]), float(line[-2]), float(line[-3])))
                        
                N = len(coors)
                xs,ys,zs = 0,0,0
                for x,y,z in coors:
                    xs += x
                    ys += y
                    zs += z
                xm,ym,zm = xs/N, ys/N, zs/N  # coordinate of molecular centroid
                mcList.append((xm,ym,zm))
                
                gen_coor = np.array([np.sqrt(x*x+y*y+z*z) for (x,y,z) in coors])
                cop = np.mean(gen_coor)  # center of position
                Rp2 = sum((gen_coor-cop)**2)/len(gen_coor) # RadGy_pos_squared, array operation
                Rg.append(np.sqrt(Rp2))
        return Rg, mcList
    
    @displayExecutionTime
    def getMolecularTrajectory(self, ss_timepoints=[], writeIt=True):
        print("Tracking the molecular trajectory...")
        simObj = self.simfileObj
        inpath = simObj.getInpath()
        outpath = simObj.getOutpath("smTracking_stat")
        
        siteInpath = inpath + "/data"
        numRuns = simObj.getNumRuns()
        siteIdList = self.getMolecularSiteIDs(siteInpath, numRuns, self.molName)
        
        Rg_stat, mcCoor_stat = [], []
        vfiles =  glob(inpath + "/viewer_files/*.txt")
        #vfiles =  glob(inpath + "/test/*.txt")
        IVF = [] # Incomplete Viewer File
        N_traj = len(vfiles)
        count_traj = 0
        for i, vfile in enumerate(vfiles):
            Rg_traj, mcList = self.getCoordinates(vfile, siteIdList)
            #print(f"Rg_traj length : {len(Rg_traj)} \t N_frame = {self.N_frame}")
            if len(Rg_traj) == self.N_frame + 1:
                count_traj += 1
                Rg_stat.append(Rg_traj)
                mcCoor_stat.append(mcList)
            else:
                IVF.append(vfile)
            ProgressBar("Progress", (i+1)/N_traj)
        
        t1000 = [t*1e3 for t in self.timeSeries]
        ss_tp1000 = [t*1e3 for t in ss_timepoints]
        
        with open(outpath + "/Sampling_stat.txt","w") as of:
            print("Complete trajectories : ", count_traj)
            of.write(f"Steady state time points (ms) : {ss_tp1000}\n\n")
            of.write(f"Number of complete trajectories : {count_traj}\n\n")
            if len(IVF) > 0:
                print("Incomplete trajectories:")
                of.write("Incomplete trajectories:\n\n")
                for traj_name in IVF:
                    name = traj_name.split('/')[-1]
                    print (name)
                    of.write(f"{name}\n\n")
        
        if self.msd_calc:
            print("Calculating the molecular diffusion....")
            SD_traj = []   # squared displacement
            dist_func_3d = lambda p1, p2: (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2
            dist_func_1d = lambda p1, p2: (p2[0]-p1[0])**2 
            
            """
            r = (x, y, z)
            mcCoor_stat = [[r_0t0, r_0t1, r_0t2, ..., r_0tn], ..., [r_nt0, r_nt1, ..., r_ntn]]
            disp = dr**2 = (xt-x0)**2 + (yt-y0)**2 + (zt-z0)**2 
            msd = mean of disp over n independent trajectories
            """
            for traj in mcCoor_stat:
                dr2_traj = []
                tn = 0
                while tn < len(traj):
                    if self.coor.upper() == '3D':
                        dr2 = dist_func_3d(traj[tn], traj[0])
                    elif self.coor.upper() == '1D':
                        dr2 = dist_func_1d(traj[tn], traj[0])
                        
                    dr2_traj.append(dr2)
                    tn += 1
                SD_traj.append(dr2_traj)
            
            MSD_traj = np.mean(np.array(SD_traj), axis=0)
            
                
            if writeIt:
                with open(outpath + f"/{self.molName}_MSD.csv","w", newline='') as tmpf:
                    wf = writer(tmpf, delimiter=',')  # csv writer
                    wf.writerow(['Time(s)','MSD(nm^2)'])
                    # timepoints have to be in 'seconds' unit in order to get correct D unit
                    wf.writerows(zip(self.timeSeries, MSD_traj))
                print("... done writing!")
             
        
        if self.radGy_calc:
            print("Calculating Radius of Gyration....")
            Rg_arr = np.array(Rg_stat)
            mean_Rg = np.mean(Rg_arr, axis=0, dtype=np.float64)
            ss_index= getSteadyStateIndicies(self.timeSeries, ss_timepoints=ss_timepoints)
            SS_Rg = Rg_arr[:,ss_index]
        
            if writeIt:
                with open(outpath + f"/{self.molName}_RadGy_dynamics.csv","w", newline='') as outfile1, open(outpath + f"/{self.molName}_RadGy_distribution.txt","w") as outfile2:
                    wf1 = writer(outfile1,delimiter=',')  # csv writer
                    wf1.writerow(['Time(ms)','RadGy(nm)'])
                    wf1.writerows(zip(t1000, mean_Rg))
                    np.savetxt(outfile2, SS_Rg, fmt="%.4e", delimiter=',')
                print("... done writing!")
            

class RunTimeStatistics:
    
    def __init__(self, txtfile, runs=None):
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
            plt.title(simName)
            plt.savefig(outpath + "/runTime.png", dpi=100)
            plt.close()
            print("Done plotting the stats ...!")
        else:
            print("Time units are different! Can't plot the runtime stats")
        
            

