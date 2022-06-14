import pandas as pd
import os
import natsort

# sdfs = os.listdir('smiles')
# sdfs = natsort.natsorted(sdfs)
# print(sdfs)
# for sdf in sdfs:
#     os.system('mol2vec featurize -i smiles/{0} -o smiles_csv/embedded_smiles.csv -m model_300dim.pkl -r 1 --uncommon UNK'.format(sdf))
# exit()

raw_data = pd.read_csv('data_new.csv')
smiles_emb = pd.read_csv('smiles_csv/embedded_smiles.csv')
data_total = pd.concat([raw_data, smiles_emb], axis=1)
data_total.to_csv('total_data.csv')