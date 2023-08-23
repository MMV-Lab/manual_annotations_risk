import pandas as pd
from sklearn.metrics import r2_score


path_napari_gt = "./csvs/napari_gt.csv"
path_slicer_gt = "./csvs/slicer_gt.csv"
path_bioGT = "./csvs/bioGT.csv"

napari_gt_df = pd.read_csv(path_napari_gt)
slicer_gt_df = pd.read_csv(path_slicer_gt)
bioGT_df = pd.read_csv(path_bioGT)

print('Coefficient of Determination volume for the Slicer-GT:', r2_score(bioGT_df['volume'],slicer_gt_df['volume']))    
print('Coefficient of Determination volume for the Napari-GT:', r2_score(bioGT_df['volume'],napari_gt_df['volume']))

# PCA of the SH coefficients
# from sklearn.decomposition import PCA
# pca = PCA(n_components=8)
# pca_bioGT = pca.fit(bioGT_df.drop(columns=['CellId', 'volume']).iloc[:].to_numpy())

# pcs_bioGT = pca_bioGT.transform(bioGT_df.drop(columns=['CellId', 'volume']).to_numpy())
# pcs_slicer_gt = pca_bioGT.transform(slicer_gt_df.drop(columns=['CellId', 'volume']).to_numpy())
# pcs_napari_gt = pca_bioGT.transform(napari_gt_df.drop(columns=['CellId', 'volume']).to_numpy())

# for i in range(pcs_bioGT.shape[1]):
#     print(f'Coefficient of Determination pc_{i} for the Slicer-GT:', r2_score(pcs_bioGT[:,i], pcs_slicer_gt[:,i]))    
#     print(f'Coefficient of Determination pc_{i} for the Napari-GT:', r2_score(pcs_bioGT[:,i], pcs_napari_gt[:,i]))  
