from cipgd.config import OUTPUT_DIR, MIN_RESPONSES
from cipgd.utils import *
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json

def analyse(qdf, Xq, out):
    qid=str(qdf['question_id'].iloc[0]); stem=safe_filename(qid); tables=ensure_dir(out/'tables'); figs=ensure_dir(out/'figures'); y=numeric_agreement(qdf)
    n=min(5, Xq.shape[1], len(qdf)-1); pcs=PCA(n_components=n, random_state=42).fit_transform(Xq)
    scores=qdf.copy(); scores['agreement_all_numeric']=y; scores['semantic_centrality']=semantic_centrality(Xq)
    for k in range(n): scores[f'pc{k+1}']=pcs[:,k]
    rows=[]
    for k in range(n):
        mask=y.notna(); rho,p=(spearmanr(pcs[mask,k], y[mask]) if mask.sum()>=5 and y[mask].nunique()>1 else (np.nan,np.nan)); rows.append({'question_id':qid,'pc':f'PC{k+1}','spearman_rho_with_agreement_all':rho,'spearman_p':p})
    pca=pd.DataFrame(rows); pca.to_csv(tables/f'question_{stem}_pca_agreement_correlations.csv', index=False)
    scores.sort_values('semantic_centrality', ascending=False).head(25).to_csv(tables/f'question_{stem}_bridge_candidates.csv', index=False)
    scores.to_csv(tables/f'question_{stem}_response_scores.csv', index=False)
    fig,ax=plt.subplots(figsize=(7,5)); sc=ax.scatter(scores.pc1, scores.pc2, c=y if y.notna().sum()>=3 else None, s=18, alpha=.75); ax.set_title(f'{qid}: PCA response map'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); fig.tight_layout(); fig.savefig(figs/f'question_{stem}_pca_map.png', dpi=200); plt.close(fig)
    return {'question_id':qid,'question_text':qdf['question_text'].iloc[0], 'n_responses':len(qdf), 'semantic_dispersion_mean_pairwise_cosine':mean_pairwise_cosine_distance(Xq), 'agreement_mean':float(np.nanmean(y)) if y.notna().any() else None, 'agreement_sd':float(np.nanstd(y)) if y.notna().any() else None, 'pca_top_abs_agreement_rho':float(pca['spearman_rho_with_agreement_all'].abs().max()) if pca['spearman_rho_with_agreement_all'].notna().any() else None, 'umap_status':'not_run'}

def main():
    df=load_response_table(OUTPUT_DIR/'tables'/'ask_opinion_response_table.parquet'); X,idx=load_embeddings(OUTPUT_DIR/'response_embeddings'/'response_embeddings.npy', OUTPUT_DIR/'response_embeddings'/'response_embedding_index.csv'); df,X=align_table_and_embeddings(df,X,idx); out=ensure_dir(OUTPUT_DIR/'question_analysis')
    summaries=[]
    for qid in choose_top_questions(df, MIN_RESPONSES):
        mask=df.question_id.astype(str)==qid; summaries.append(analyse(df.loc[mask].reset_index(drop=True), X[mask.to_numpy()], out))
    pd.DataFrame(summaries).to_csv(out/'tables'/'all_question_summary.csv', index=False); print('Saved question analysis')
if __name__=='__main__': main()
