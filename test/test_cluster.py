import cluster
import siteio
import os
import pandas as pd
import random
import makesim
import itertools
import numpy as np

def test_similarity():
    '''
    Similarity tests that the function returns a similarity value. If the same active 
    site is fed in, it's similarity with itself is asserted to be 1. If a different
    active site is fed in, similarity is between 0 and 1.
    is 1. 
    '''
    files = [f for f in os.listdir('sample_active_sites/') if f.endswith('.pdb')]
    pathtomat = 'sample_active_sites/12859_2009_3124_MOESM2_ESM.norm.csv'
    mat = pd.read_csv(pathtomat,index_col=0)
    k = 10
    n = 0.5
    s = 0.5

    # same active sites
    for a in range(len(files)):
        filename_a = os.path.join("sample_active_sites", files[a])

        activesite_a = siteio.read_active_site(filename_a)
        activesite_b = siteio.read_active_site(filename_a)

        sim = makesim.compute_similarity(activesite_a, activesite_b, mat, k, n, s)
        assert sim == 1

    # different active sites
    for a in range(len(files)):
        for b in range(a+1,len(files)):
            filename_a = os.path.join("sample_active_sites", files[a])
            filename_b = os.path.join("sample_active_sites", files[b])

            activesite_a = siteio.read_active_site(filename_a)
            activesite_b = siteio.read_active_site(filename_b)

            sim = makesim.compute_similarity(activesite_a, activesite_b, mat, k, n, s) 
            assert sim >= 0 and sim  <= 1

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    simmat = pd.read_pickle('sim.300.50.70.pkl')
    k = 2
    M, C, sitedict = cluster.cluster_by_partitioning(pdb_ids,simmat,k=k)
    assert len(M) == k and len(C) == k and len(sitedict) == len(pdb_ids)

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    # active_sites = []
    # for id in pdb_ids:
    #     filepath = os.path.join("active_sites", "%i.pdb"%id)
    #     active_sites.append(siteio.read_active_site(filepath))
    # update this assertion
    
    simmat = pd.read_pickle('sim.300.50.70.pkl')

    Z = cluster.cluster_hierarchically(pdb_ids, simmat)

# C = test_partition_clustering()
# Z = test_hierarchical_clustering()
# print(C)
# print(Z)