import pandas as pd
import glob, os

base = "D:/Developer/NOMA_new/dataset/results/"
folders = [f for f in glob.glob(os.path.join(base, "results_*"))]

all_users, all_pairs = [], []
for folder in folders:
    h_df = pd.read_csv(os.path.join(folder, "h_values.csv"))
    c_df = pd.read_csv(os.path.join(folder, "bipartite_pf_clustering.csv"))
    h_df['Graph_ID'] = folder.split('_')[-1]
    c_df['Graph_ID'] = folder.split('_')[-1]
    all_users.append(h_df)
    all_pairs.append(c_df)

users_df = pd.concat(all_users)
pairs_df = pd.concat(all_pairs)
users_df.to_csv("merged_h_values.csv", index=False)
pairs_df.to_csv("merged_pairs.csv", index=False)
