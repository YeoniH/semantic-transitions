from cipgd.config import OUTPUT_DIR
from cipgd.utils import *
from sklearn.metrics import pairwise_distances
import pandas as pd, numpy as np

def main():
    df=load_response_table(OUTPUT_DIR/'tables'/'ask_opinion_response_table.parquet'); X,idx=load_embeddings(OUTPUT_DIR/'response_embeddings'/'response_embeddings.npy', OUTPUT_DIR/'response_embeddings'/'response_embedding_index.csv'); df,X=align_table_and_embeddings(df,X,idx); out=ensure_dir(OUTPUT_DIR/'cross_group'/'tables'); df['_pos']=np.arange(len(df)); y=numeric_agreement(df); gc=X.mean(axis=0,keepdims=True); rows=[]
    for col in available_group_columns(df):
        for group,g in df.groupby(col, dropna=False):
            if len(g)<50: continue
            Xg=X[g._pos.to_numpy()]; rows.append({'group_column':col,'group_value':str(group),'n_responses':len(g),'mean_agreement_all':y.loc[g.index].mean(),'semantic_dispersion_mean_pairwise_cosine':mean_pairwise_cosine_distance(Xg),'distance_to_global_centroid':pairwise_distances(Xg.mean(axis=0,keepdims=True),gc,metric='cosine')[0,0]})
    pd.DataFrame(rows).to_csv(out/'all_group_summaries.csv', index=False); print(f'Saved {out}')
if __name__=='__main__': main()
