from cipgd.config import OUTPUT_DIR
from cipgd.utils import *
from sklearn.metrics import pairwise_distances
import pandas as pd, numpy as np

def main():
    df=load_response_table(OUTPUT_DIR/'tables'/'ask_opinion_response_table.parquet'); X,idx=load_embeddings(OUTPUT_DIR/'response_embeddings'/'response_embeddings.npy', OUTPUT_DIR/'response_embeddings'/'response_embedding_index.csv'); df,X=align_table_and_embeddings(df,X,idx); out=ensure_dir(OUTPUT_DIR/'trajectories'/'tables'); df['_pos']=np.arange(len(df)); rows=[]
    for pid,g in df.dropna(subset=['participant_id']).groupby('participant_id'):
        g=g.sort_values('response_row_id').reset_index(drop=True)
        if len(g)<2: continue
        Xg=X[g._pos.to_numpy()]
        for i in range(len(g)-1): rows.append({'participant_id':pid,'step':i,'from_question_id':g.loc[i,'question_id'],'to_question_id':g.loc[i+1,'question_id'],'cosine_semantic_shift':pairwise_distances(Xg[i:i+1],Xg[i+1:i+2],metric='cosine')[0,0]})
    pd.DataFrame(rows).to_csv(out/'participant_semantic_shifts.csv', index=False); print(f'Saved {out}')
if __name__=='__main__': main()
