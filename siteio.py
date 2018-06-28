import os
import Bio.PDB

def read_active_site(filepath):
    """
    Read in a single active site given a PDB file

    Input: PDB file path
    Output: ActiveSite instance
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] != ".pdb":
        raise IOError("%s is not a PDB file"%filepath)
    
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    active_site = pdb_parser.get_structure(id='actsite',file=filepath)
    return active_site