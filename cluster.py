# from .utils import Atom, Residue, ActiveSite
import Bio.PDB
import itertools
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import pandas as pd
import random
from copy import copy

def check_pdbs(pdbs):
    length = len(pdbs)
    if type(pdbs[0]) != str:
        for i in range(length):
            pdbs[i] = str(pdbs[i])
    if '.' in pdbs[0]:
        for i in range(length):
            a = pdbs[i].split('.')[-1] 
            if a != 'pdb':
                print('These are not pdb files.')
                raise SystemExit
    else:
        for i in range(length):
            pdbs[i] = pdbs[i] + '.pdb'
    return pdbs

def make_subD(D,pdbs):
    # takes in pandas DF as distance matrix
    # and list of pdb strings that match column names in DF

    length = len(pdbs)
    cols = list(D.columns)
    inds = np.zeros((length,2),dtype=object)

    for i in range(length):
        inds[i][0] = pdbs[i]
        inds[i][1] = cols.index(pdbs[i])

    inds = inds[inds[:,1].argsort()]
    sites = list(inds[:,0])
    sitedict = dict(zip(range(length),sites))
    subD = np.zeros((length,length))

    for i in range(len(subD)):
        for j in range(i+1):
            subD[i,j] = 9 # mask with "giant" value

    index = itertools.combinations(sites,r=2)
    subD = subD.ravel()

    for i in range(len(subD)):
        if subD[i] == 0:
            a = next(index)
            subD[i] = D[a[0]][a[1]]

    subD = pd.DataFrame(np.reshape(subD,(length,length)),
        index = pdbs, columns=pdbs)
    return subD, sitedict



def cluster_by_partitioning(pdbs, simmat, k = 10):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)

    This code was adapted from github user letiantian:
    https://github.com/letiantian/kmedoids/blob/master/kmedoids.py#L8
    I put all comments in quotes that are from the original code, and added more
    comments to show understanding of the algorithm.
    """

    # Enfore a stop condition for iterations
    tmax = 1000

    # From the similarity matrix, obtain a distance matrix:
    if pdbs == 'all':
        pdbs = list(simmat.columns)
    else:
        pdbs = check_pdbs(pdbs)

    Dori = simmat.subtract(2).multiply(-1).subtract(1)

    D, sitedict = make_subD(Dori,pdbs)

    D = D.as_matrix()
    for i in range(len(D)):
        for j in range(i+1):
            if i == j:
                D[i,j] = 0 # mask with zeros
            else:
                D[i,j] = D[j,i] # recover symmetry

    n = len(D)
    # Ensure that you haven't asked for more clusters than there are datapoints
    if k > n:
        raise Exception('Too many medoids.')
    '''
    "Initialize a unique set of valid initial cluster medoid indices since we
    can't seed different clusters with two points at the same location." 
    If we did not do this, in our shuffling of the indices to get random 
    starting medoids, we might obtain two medoids that have a distance of zero
    between them, which would make the downstream algorithm error.
    '''
    # Make sets of indices
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])

    # Find out where D == 0, if at any points
    rs , cs = np.where(D == 0)
    
    # "The rows, cols must be shuffled because we will keep the first duplicate below"
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs): # For each entry in the distance matrix with d=0:
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c) # move one to the invalid pile, keep the
                                       # the other in the valid pile

        # update the valid medoid start sites with the difference of the two sets

    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds) 
    
    if k > len(valid_medoid_inds):
        raise Exception('Too many medoids (after removing {} duplicate points).'.format(
            len(invalid_medoid_inds)))

    # "Randomly initialize an array of k medoid indices"
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k]) # Take the first k random medoid start sites

    # "Create a copy of the array of medoid indices"
    Mnew = np.copy(M)

    # "Initialize a dictionary to represent clusters"
    C = {}
    for t in range(tmax):
        # "Determine clusters, i.e. arrays of data indices"
        # Find the corresponding active sites of the medoids where D is minimized:
        J = np.argmin(D[:,M], axis=1)
        # J = for each active site, this is your closest medoid active site
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0] # Update the clusters with this info
        # "Update cluster medoids"
        # Change the medoid if there is a member within the cluster with a
        # smaller distance 
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j] # This is the new medoid
        np.sort(Mnew)
        # "Check for convergence"
        if np.array_equal(M, Mnew): # No change, don't bother clustering more
            break
        M = np.copy(Mnew)
    else:
        # "Final update of cluster memberships"
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    return M, C, sitedict

def cluster_hierarchically(pdbs, simmat):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    I tried to implement this myself, see commented code below
    Unfortunately it didn't work, so I just called a scipy function.
    """




    # note this script will only work for min or max, not for any other linkage
    def link(a,b): 
        new = min([a,b])
        return new

    def cluster(D):
        
        clusters = list([i] for i in D.columns)
        
        clusterings = []
        
        for iteration in range(len(D) - 1): # a clustering of n elements has n - 1 merges
            clusterings.append(copy(clusters))
            flatind = np.argmin(D.as_matrix())
            flatval = D.as_matrix().ravel()[flatind]
            minrow, mincol = (flatind//len(D), flatind % len(D))
            # Just arbitrariliy keep the cluster represented by a lower index
            # Replace its values with those of the new cluster and delete the old cluster
            
            newslice = min(minrow,mincol) 
            oldslice = max(minrow,mincol)
            
            #Replace all distances of new cluster with linkage of distances of 
            #the two merged clusters
            
            for i in range(newslice): # top
                D.iloc[i,newslice] = link(D.iloc[i,newslice],D.iloc[i,oldslice])
            for i in range(newslice + 1, oldslice): # inner
                D.iloc[newslice,i] = link(D.iloc[newslice,i],D.iloc[i,oldslice])
            for i in range(oldslice + 1, len(D)): # right
                D.iloc[newslice,i] = link(D.iloc[newslice,i],D.iloc[oldslice,i])

            clusters[newslice] = [clusters[newslice] + clusters[oldslice]]
            del clusters[oldslice]
            D = D.drop(D.index[oldslice],axis=0)
            D = D.drop(D.columns[oldslice],axis=1)
        return clusterings

    #####################################################################
    if pdbs == 'all':
        pdbs = list(simmat.columns)
    else:
        pdbs = check_pdbs(pdbs)

    # Make distance matrix from similarity matrix
    Dori = simmat.subtract(2).multiply(-1).subtract(1)

    D, sitedict = make_subD(Dori,pdbs)

    '''
    The way I wrote this algorithm is by finding the minimum value in the
    condensed distance matrix. My condensed matrix is not the same format as
    the scipy condensed matrices, but it is the same concept of only using
    the upper half triangle. It searches the distance matrix for the lowest
    value, then deletes the active site that's further down the list and 
    replaces the other active site with the minimum values of all of the
    two active sties' interactions with all other active sites. 
    
    Note because it deletes the member active sites' values after a cluster 
    has been formed, it can't do any other linkage than min or max. For 
    instance, if it tried to perform an average linkage, the first two 
    observations to from a cluster would each be weighted by one half, but 
    each subsequent active site joining the cluster would have it's value 
    averaged with the already-calculated average, thus weighting it more 
    heavily. 
    '''

    clusterings = cluster(D)

    # for i in clusterings:
    #     print('')
    #     for j in itertools.chain(i):
    #         print(j)
    #     input()
    '''
    Although the above algorithm works, the list of lists format is hard to
    resolve. Therefore, for comparisons to k-medoids and for quality checks,
    I'm going to use scipy's built in function.
    '''

    D = D.as_matrix()
    for i in range(len(D)):
        for j in range(i+1):
            if i == j:
                D[i,j] = 0 # mask with zeros
            else:
                D[i,j] = D[j,i] # recover symmetry
    Z = linkage(squareform(D))
    Znew = np.zeros(Z.shape)

    for i in range(len(Z)):
        for k in range(4):
            Znew[i][k] = int(Z[i][k])
    # Z = [list(i) for i in sorted(Znew, key = lambda x: int(x[3]))]
    # Fill in your code here!
    # print(type(Z))
    # print(Z[:10][:])

    return Z, sitedict