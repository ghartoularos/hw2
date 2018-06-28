import os
import pandas as pd
import numpy as np

def normalize_matrix(mat):
    mat = mat.subtract(min(mat.min()))
    mat = mat.divide(max(mat.max()))
    matcols = mat.divide(mat.max())
    matrows = mat.divide(mat.max(0),0)
    mat = pd.DataFrame(np.zeros((len(mat),len(mat))),
        index=mat.index,
        columns=mat.columns)
    for i in range(len(mat)):
        for j in range(len(mat)):
            mat.iloc[i,j] = np.mean([matcols.iloc[i,j],
                                    matrows.iloc[i,j]])
    return mat

if __name__ == '__main__':
    mat = pd.read_csv(os.path.join("active_sites", "12859_2009_3124_MOESM2_ESM.mat"),sep ='\s')
    mat = normalize_matrix(mat)
    mat.to_csv(os.path.join("active_sites", "12859_2009_3124_MOESM2_ESM.norm.csv"))