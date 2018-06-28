import argparse
import os
import pandas as pd
import numpy as np
import siteio
import Bio.SeqUtils
convert = Bio.SeqUtils.IUPACData.protein_letters_3to1
from scipy.spatial.distance import pdist

def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict

convert = _uppercase_for_dict_keys(convert)

def seq_simlarity(seq1, seq2, length, mat, seqdict=None):
    score = 0
    for i in range(length):
        score += mat.loc[seq1[i],seq2[i]]
    normscore = float(score)/float(length)

    if seqdict != None:
        seqdict.update({(seq1,seq2): normscore})
        seqdict.update({(seq2,seq1): normscore})
        return float(score)/float(length), seqdict
    else:
        return float(score)/float(length)

def norm_rmsd(site1, site2, n, length):
    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(site1, site2)
    super_imposer.apply(site2)

    # RMSD:
    rmsd = super_imposer.rms
    site1vecs = np.zeros((length,3))
    site2vecs = np.zeros((length,3))
    row = 0
    for i in site1:
        site1vecs[row] = i.coord
    for i in site2:
        site2vecs[row] = i.coord

    maxrmsd = max(np.amax(pdist(site1vecs,'euclidean')),
        np.amax(pdist(site2vecs,'euclidean')))
    normrmsd = 1 - float(rmsd)/(float(maxrmsd)*n)
    if normrmsd < 0:
        normrmsd = 0
    elif normrmsd > 1:
        normrmsd = 1
    return normrmsd

def compute_similarity(site_a, site_b, mat, 
                       k=10, n=0.5, s=0.7):
    '''
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)

    Some parameters to play with:

    *k*
    increasing this increases accuracy, more random iterations 
    so better chance of finding optimal order

    *n*
    increasing this increases chances of attaining high structural similarity
    
    *s*
    increasing this increases contirbution of sequence similarity
    to overall score
    '''

    # Use the first model in the pdb-files for alignment
    site_a = site_a[0]
    site_b = site_b[0]
    
    # Iterate over all chains in the model in order to get residues
    a_atoms = []
    b_atoms = []

    # Pull out the CA atoms from Biopython's PDB structure object
    for chain in site_a:
        for res in chain:
            a_atoms.append(res['CA'])
    for chain in site_b:
        for res in chain:
            b_atoms.append(res['CA'])

    #trivial check:
    if len(a_atoms) == len(b_atoms): #  these might be the same site
        seq1 = ''.join([convert[j.parent.resname] for j in a_atoms])
        seq2 = ''.join([convert[j.parent.resname] for j in b_atoms])
        if (norm_rmsd(a_atoms,b_atoms,n=1,length=len(a_atoms)) > 0.999 and 
            seq_simlarity(seq1, seq2, len(a_atoms), mat) == 1.0):
            return 1
    # Get number of residues in each active site
    alen = len(a_atoms)
    blen = len(b_atoms)
    length = min(alen,blen) # Going to standaridize to site with less residues

    # Initializations
    seqdict = dict()
    sims = list()
    tupes = list()

    '''
    This is where the randomness comes in. Because the order of the residues
    listed in any given active site is irrelevant, but rmsd and sequence
    similarity both require some inherent order for scoring, I've taken random
    combinations of the the list of atoms for each given active site.
    '''
    if alen == blen:
        for _ in range(k):
            set1 = np.random.permutation(a_atoms)
            set2 = np.random.permutation(b_atoms)
            tupes.append((set1, set2))
    elif alen > blen:
        for _ in range(k):
            set1 = np.random.choice(a_atoms,len(b_atoms))
            set2 = np.random.permutation(b_atoms)
            tupes.append((set1, set2))
    else:
        for _ in range(k):
            set1 = np.random.permutation(a_atoms)
            set2 = np.random.choice(b_atoms,len(a_atoms))
            tupes.append((set1, set2))

    for i in tupes:
        seq1 = ''.join([convert[j.parent.resname] for j in i[0]])
        seq2 = ''.join([convert[j.parent.resname] for j in i[1]])
        try:
            # print(seq1,seq2)
            seqsim = seqdict[(seq1, seq2)]
        except:
            seqsim, seqdict = seq_simlarity(seq1, seq2, length, mat, seqdict)
        structsim = norm_rmsd(i[0],i[1], n, length)
        sims.append(np.average([seqsim, structsim],weights=(s,(1 - s))))

    sim = max(sims)

    return sim

if __name__ == "__main__":
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="""
    Generate the similarity matrix.
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', metavar='pathtomat', type=str, 
                        default='active_sites/12859_2009_3124_MOESM2_ESM.norm.csv',
                        help='path to normalized electrostatic similarity csv file')

    parser.add_argument('-k', metavar='perms', type=int,  default=10,
                        help='number of times to randomly permute atoms [def: 10]')

    parser.add_argument('-n',metavar='norm',type=float, default=0.5,
                        help='normalization factor for RMSD')

    parser.add_argument('-s', metavar='simweight', type=float, default=0.7,
                    help='weight for sequence similarity [def: 0.7]')


    args = parser.parse_args()  # Parse arguments
    pathtomat = args.p
    k = args.k
    n = args.n
    s = args.s

    mat = pd.read_csv(pathtomat,index_col=0)
    files = [f for f in os.listdir('active_sites/') if f.endswith('.pdb')]
    simmat = pd.DataFrame(np.zeros((len(files),len(files))),columns=files,index=files)

    for a in tqdm(range(len(files))):
        for b in range(a,len(files)):
            filename_a = os.path.join("active_sites", files[a])
            filename_b = os.path.join("active_sites", files[b])

            activesite_a = siteio.read_active_site(filename_a)
            activesite_b = siteio.read_active_site(filename_b)

            sim = compute_similarity(activesite_a, activesite_b, mat, k, n, s)

            simmat.iloc[a,b] = sim
            simmat.iloc[b,a] = sim
    simmat.to_pickle('.'.join(['sim',str(k),str(int(100*n)),str(int(100*s)),'pkl']))