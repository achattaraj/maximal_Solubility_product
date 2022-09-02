# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:43:43 2021

@author: Ani Chattaraj
"""

from DataPy import ReadInputFile, InterSiteDistance, ProgressBar, displayExecutionTime
import re, json, pickle
import numpy as np
import networkx as nx
from glob import glob
import matplotlib.pyplot as plt
from csv import writer
from numpy import pi, array
from collections import defaultdict, OrderedDict, namedtuple, Counter

font = {'family' : 'Arial',
        'size'   : 16}

plt.rc('font', **font)


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)
# pos = center of mass, radius = Radius of gyration, 
#density = sites/volume
Cluster = namedtuple('Cluster', ['pos', 'radius', 'density'])

class ClusterDensity:

    def __init__(self, txtfile, ss_timeSeries):
        self.simObj = ReadInputFile(txtfile)
        tf, dt, dt_data, dt_image = self.simObj.getTimeStats()
        self.ss_timeSeries = ss_timeSeries
        inpath = self.simObj.getInpath() + "/data"
        numRuns = self.simObj.getNumRuns()
        self.N_frames = int(tf/dt_image)
        self.inpath = inpath
        if numRuns == 0:
            self.numRuns = 25
        else:    
            self.numRuns = numRuns

    def __repr__(self):
        simfile = self.simObj.txtfile.split('/')[-1]
        info = f"Class : {self.__class__.__name__}\nSystem : {simfile}"
        return info

    @staticmethod
    def getMolIds(molfile, molName):
        molDict = {}
        mIds = []
        with open(molfile, 'r') as tmpfile:
            for line in tmpfile:
                line = line.strip().split(',')
                if line[-1] == molName:
                    mIds.append(line[0])
        molDict[molName] = mIds
        return molDict

    @staticmethod
    def getSiteIds(sitefile, molName):
        siteDict = {}
        sIds = []
        with open(sitefile, 'r') as tmpfile:
            for line in tmpfile:
                line = line.strip().split(',')
                if line[-1].split()[0] == molName:
                    sIds.append(line[0])
        siteDict[molName] = sIds
        return siteDict

    @staticmethod
    def getRevDict(myDict):
        # k,v = key : [val1, val2, ...]
        revDict = {}
        for k,v in myDict.items():
            for val in v:
                revDict[val] = k
        return revDict
    @staticmethod
    def splitArr(arr, n):
        subArrs = []
        if (len(arr)%n == 0):
            f = int(len(arr)/n)
            i = 0
            while (i < len(arr)):
                sub_arr = arr[i : i+f]
                i += f
                subArrs.append(sub_arr)
            return subArrs
        else:
            print(f"Can't split the given array (length = {len(arr)}) into {n} parts")

    def mapSiteToMolecule(self):
        sIDfile = InterSiteDistance.findFile(self.inpath, self.numRuns, "SiteIDs")
        mIDfile = InterSiteDistance.findFile(self.inpath, self.numRuns, "MoleculeIDs")
        molName, molCount = self.simObj.getMolecules()
        molIDs, siteIDs = {}, {}
        for mol in molName:
            molIDs.update(self.getMolIds(mIDfile, mol))
            siteIDs.update(self.getSiteIds(sIDfile, mol))
        spmIDs = {} # sites per molecule
        for mol, count in zip(molName, molCount):
            #molDict = {}
            arr = self.splitArr(siteIDs[mol], count)
            mol_ids = molIDs[mol]
            d = dict(zip(mol_ids, arr))
            spmIDs.update(d)
        rev_spm = self.getRevDict(spmIDs)
        return rev_spm

    @staticmethod
    def getBindingStatus(frame):
        linkList = []
        posDict = {}
        for curline in frame:
            if re.search("ID", curline):
                line = curline.split()
                posDict[line[1]] = [float(line[4]),float(line[5]), float(line[6])] # order = x,y,z
                #IdList.append(line.split()[1])

            if re.search("Link", curline):
                line = curline.split()
                linkList.append((line[1], line[3]))
        return posDict, linkList

    @staticmethod
    def createGraph(IdList, LinkList):
        G = nx.Graph()
        G.add_nodes_from(IdList)
        G.add_edges_from(LinkList)
        return G
    
    @staticmethod
    def createMultiGraph(LinkList):
        MG = nx.MultiGraph()
        MG.add_edges_from(LinkList)
        return MG

    @staticmethod
    def getFrameIndices(viewerfile):
        frame_indices = []
        tps = []
        with open(viewerfile, 'r') as tmpfile:
            lines = tmpfile.readlines()
            for i, line in enumerate(lines):
                if re.search("SCENE", line):
                    frame_indices.append(i)
                    tp = lines[i+1].split()[-1]
                    tps.append(tp)

            frame_indices.append(len(lines))

        return tps, frame_indices
    
    def getSteadyStateFrameIndices(self, viewerfile):
        frame_indices = []
        ss_indices = [] 
        tps = []
        index_pairs = []
        with open(viewerfile, 'r') as tmpfile:
            lines = tmpfile.readlines()
            for i, line in enumerate(lines):
                if re.search("SCENE", line):
                    frame_indices.append(i)
                    tp = lines[i+1].split()[-1]
                    tps.append(tp)
                    if any([np.isclose(float(tp), t) for t in self.ss_timeSeries]):
                        ss_indices.append(i)
            frame_indices.append(len(lines))
        for ii, elem in enumerate(frame_indices):
            if elem in ss_indices:
                index_pairs.append((elem, frame_indices[ii+1]))
        return tps, index_pairs

    @staticmethod
    def calc_RadGy(posList):
        # posList = N,3 array for N sites
        com = np.mean(posList, axis=0) # center of mass
        Rg2 = np.mean(np.sum((posList - com)**2, axis=1))
        return com, np.sqrt(Rg2)
    
    @staticmethod
    def calc_zagreb_indices(MG):
        # MG: MULTI-GRAPH Object # multiple edges allowed between two nodes 
        d1List = []
        d2List = []
        nodes = list(MG.nodes())
        #links = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(len(nodes)) if i<j]
        for n in nodes:
            d1List.append(MG.degree(n))
        for n1,n2 in set(MG.edges()):
            d2List.append((MG.degree(n1), MG.degree(n2)))
        d1Arr = array(d1List)
        
        # M1, M2: first and second Zagreb indicies
        M1 = sum(d1Arr**2)
        M2 = sum([d1*d2 for d1,d2 in d2List])
        
        return M1, M2, d1Arr
        

    def getClusterDensity(self, viewerfile, cs_thresh=1):
        # cluster size,  radius of gyration
        # M1, M2: Zagreb indices
        csList, RgList, M1List, M2List = [], [], [], []
        
        mtp_cs, mtp_rg = [], [] # mtp: multi timepoint stat
        
        # MCL: molecular cross linking (number of bonds per molecule)  
        MCL = []
        msm = self.mapSiteToMolecule()
        tps, index_pairs = self.getSteadyStateFrameIndices(viewerfile)
         
        with open(viewerfile, 'r') as infile:
            lines = infile.readlines()
            for i,j in index_pairs:
                #i,j = index_pairs[-1]
                current_frame = lines[i:j]
                cs_frame, rg_frame = [], [] # clusters in current frame
                posDict, Links = self.getBindingStatus(current_frame)
                Ids = [_ for _ in posDict.keys()]
                #mIds, mLinks = [msm[k] for k in Ids], [(msm[k1], msm[k2]) for k1,k2 in Links]
                sG = self.createGraph(Ids, Links)
                #mG = self.createGraph(mIds, mLinks)
                     
                #G.subgraph(c) for c in connected_components(G)
                for sg in connected_component_subgraphs(sG):
                    mLinks = [(msm[k1], msm[k2]) for k1, k2 in sg.edges()]
                    # connection between two different molecules
                    bonds = [(m1,m2) for m1,m2 in mLinks if m1 != m2]
                     
                    MG = self.createMultiGraph(bonds)
                    # cluster size (number of molecules)
                    cs = len(MG.nodes())
                    
                    if cs > cs_thresh:
                        M1, M2, dArr = self.calc_zagreb_indices(MG)
                        MCL.extend(dArr)
                        sites = list(sg.nodes)
                        
                        posList = np.array([posDict[s] for s in sites])
                        
                        com, Rg = self.calc_RadGy(posList)
                        
                        cs_frame.append(cs)
                        rg_frame.append(Rg)
                        
                        csList.append(cs)
                        RgList.append(Rg)
                        M1List.append(M1)
                        M2List.append(M2)
                
                mtp_cs.append(cs_frame)
                mtp_rg.append(rg_frame)
                        
       
        return [csList, RgList, M1List, M2List], MCL, mtp_cs, mtp_rg
    
    @staticmethod  
    def plotRg(csList, RgList):
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(csList, RgList, color='k', s=4)
        ax.set_xlabel('Cluster size (molecules)')
        ax.set_ylabel('Radius of Gyration (nm)')
        plt.show() 
    
    @staticmethod
    def plotBondsPerMolecule(countDict):
        fig, ax = plt.subplots(figsize=(5,3))
        bonds, freq = countDict.keys(), countDict.values()
        ax.bar(bonds, freq, width=0.3, color='grey')
        ax.set_xlabel('Bonds per molecule')
        ax.set_ylabel('Frequency')
        plt.show()
        
    @displayExecutionTime
    def getCD_stat(self, cs_thresh=1):
        # collect statistics at the last timepoint
        sysName = self.inpath.split('/')[-2].replace('_SIM_FOLDER','')
        print('\nSystem: ', sysName)
        print("Calculating Cluster Density ...")
      
        outpath = self.simObj.getOutpath("BF_stat")
        vfiles = glob(self.simObj.getInpath() + "/viewer_files/*.txt")[:]

        N_traj = len(vfiles)

        header = 'Cluster size, Rg (nm), M1, M2'
        MCL_stat = []
        cs_tmp, rg_tmp = [], []

        for i, vfile in enumerate(vfiles):
            res, MCL, mtp_cs, mtp_rg = self.getClusterDensity(vfile, cs_thresh=cs_thresh)
            #print(array(mtp_rg))
            MCL_stat.extend(MCL)
            cs_tmp.extend(mtp_cs)
            rg_tmp.extend(mtp_rg)
            
            if cs_thresh == 1:
                runNum = vfile.split('_')[-1] # Run0.txt
                #np.savetxt(outpath + '/cs_' + runNum , array(mtp_cs))
                with open(outpath + '/cs_' + runNum, 'w') as of:
                    for item in mtp_cs:
                        of.write(f'{item}')
                        of.write('\n')
                
                with open(outpath + '/rg_' + runNum, 'w') as of:
                    for item in mtp_rg:
                        of.write(f'{item}')
                        of.write('\n')
             
            ProgressBar("Progress", (i+1)/N_traj)
        
        
        counts_norm = {k: (MCL_stat.count(k)/len(MCL_stat)) for k in set(MCL_stat)}
        
        if cs_thresh == 1:
            fName = '/Bonds_per_single_molecule.csv'
        else:
            fName = f'/Bonds_per_single_molecule_cs_gt_{cs_thresh}.csv'
            
        with open(outpath + fName, "w", newline='') as of:
            obj = writer(of)
            obj.writerow(['BondCounts','frequency'])
            obj.writerows(zip(counts_norm.keys(), counts_norm.values()))
        
        csList = np.concatenate(cs_tmp).ravel().tolist()
        rgList = np.concatenate(rg_tmp).ravel().tolist()
        self.plotRg(csList, rgList)
        self.plotBondsPerMolecule(counts_norm)
       
'''
files = glob('C:/Users/chatt/Desktop/pytest/springsalad/test_dataset/A5_B5_flex_3nm_2nm_count_40_SIM_FOLDER/A5_B5_flex_3nm_2nm_count_40_SIM.txt')        

for txtfile in files[:]:
    cd = ClusterDensity(txtfile, ss_timeSeries=[ 0.02, 0.04])
    cd.getCD_stat(cs_thresh=1)
    
'''





