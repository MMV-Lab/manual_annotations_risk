import pandas as pd
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

method = "StarDist-3D"

path_wts = method + "/wts.csv"
path_man = method + "/man.csv"
path_gt = method + "/gt.csv"

wts_df = pd.read_csv(path_wts)
man_df = pd.read_csv(path_man)
gt_df = pd.read_csv(path_gt)

print('Coefficient of Determination volume for the Napari-GT model:', r2_score(gt_df['volume'],man_df['volume']))    
print('Coefficient of Determination volume for the bioGT model:', r2_score(gt_df['volume'],wts_df['volume']))

# PCA of the SH coefficients
pca = PCA(n_components=8)
pca_gt = pca.fit(gt_df.drop(columns=['CellId', 'volume', 'height']).iloc[:].to_numpy())

pcs_gt = pca_gt.transform(gt_df.drop(columns=['CellId', 'volume', 'height']).to_numpy())
pcs_man = pca_gt.transform(man_df.drop(columns=['CellId', 'volume', 'height']).to_numpy())
pcs_wts = pca_gt.transform(wts_df.drop(columns=['CellId', 'volume', 'height']).to_numpy())


for i in range(pcs_gt.shape[1]):
    print(f'Coefficient of Determination pc_{i} for the Napari-GT model:', r2_score(pcs_gt[:,i], pcs_man[:,i]))    
    print(f'Coefficient of Determination pc_{i} for the bioGT model:', r2_score(pcs_gt[:,i], pcs_wts[:,i]))  

import pdb; pdb.set_trace()