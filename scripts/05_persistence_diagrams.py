from cipgd.config import *
from cipgd.utils import *
from sklearn.metrics import pairwise_distances
import pandas as pd, numpy as np, matplotlib.pyplot as plt, textwrap

def main():
    from ripser import ripser
    df=load_response_table(OUTPUT_DIR/'tables'/'ask_opinion_response_table.parquet'); X,idx=load_embeddings(OUTPUT_DIR/'response_embeddings'/'response_embeddings.npy', OUTPUT_DIR/'response_embeddings'/'response_embedding_index.csv'); df,X=align_table_and_embeddings(df,X,idx)
    ranked_dir=OUTPUT_DIR/'question_analysis'/'ranked_views'; out=ensure_dir(OUTPUT_DIR/'topology_ranked_pd'); rows=[]
    figout=ensure_dir(out/'figures'); panelout=ensure_dir(out/'panels'); tableout=ensure_dir(out/'tables')
    for cat in RANKED_CATEGORIES:
        p=ranked_dir/f'{cat}.csv'
        if not p.exists(): continue
        items=[]; catout=ensure_dir(figout/cat)
        for rank,row in enumerate(pd.read_csv(p).head(10).itertuples(index=False), start=1):
            qid=str(row.question_id); qtext=str(row.question_text); mask=df.question_id.astype(str)==qid; qdf=df.loc[mask].reset_index(drop=True); Xq=X[mask.to_numpy()]
            if len(qdf)>MAX_POINTS_PER_QUESTION:
                rng=np.random.default_rng(RANDOM_SEED); pos=rng.choice(len(qdf), MAX_POINTS_PER_QUESTION, replace=False); qdf=qdf.iloc[pos]; Xq=Xq[pos]
            dgms=ripser(pairwise_distances(Xq, metric='cosine'), distance_matrix=True, maxdim=MAX_HOMOLOGY_DIMENSION)['dgms']; items.append((rank,qid,qtext,dgms,len(qdf)))
            for dim,dgm in enumerate(dgms):
                fig,ax=plt.subplots(figsize=(5,5)); stats=draw_pd(ax,dgm,dim,PERSISTENCE_XLIM,PERSISTENCE_YLIM,True); ax.set_title(f'PD{dim} | rank {rank} | {cat} | n={len(qdf)}\nQ: {textwrap.fill(qtext,70)}', fontsize=8); ax.set_xlabel('Birth'); ax.set_ylabel('Death'); fig.tight_layout(); fname=f"{CATEGORY_ABBREVIATIONS.get(cat,cat)}_PD{dim}_{rank}_{safe_filename(qid)}.png"; fig.savefig(catout/fname, dpi=240); plt.close(fig); rows.append({'ranked_category':cat,'rank_number':rank,'question_id':qid,'question_text':qtext,'dimension':dim,'n_points_used':len(qdf),**stats})
    pd.DataFrame(rows).to_csv(tableout/'ranked_persistence_diagram_summary.csv', index=False); print(f'Saved {out}')
if __name__=='__main__': main()
