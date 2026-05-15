from cipgd.config import OUTPUT_DIR
from cipgd.utils import ensure_dir, safe_filename
from scipy.stats import spearmanr
import pandas as pd, numpy as np, matplotlib.pyplot as plt, textwrap
METRICS=['max_finite_lifetime','mean_finite_lifetime','n_finite_features']
DEFINITION='Candidate turning points are questions with high semantic dispersion, high PCA/agreement alignment, and high agreement variability. This script checks whether they also show longer H0/H1 persistence.'
def main():
    ranked=pd.read_csv(OUTPUT_DIR/'question_analysis'/'ranked_views'/'all_question_summary_enriched.csv'); topo=pd.read_csv(OUTPUT_DIR/'topology_ranked_pd'/'tables'/'ranked_persistence_diagram_summary.csv'); m=topo.merge(ranked,on='question_id',how='left'); out=ensure_dir(OUTPUT_DIR/'topology_ranked_pd'/'verification'); plots=ensure_dir(out/'plots'); tables=ensure_dir(out/'tables'); (out/'operational_definition_candidate_turning_points.md').write_text(DEFINITION+'\n')
    m.to_csv(tables/'topology_with_question_metrics.csv', index=False); rows=[]
    for cat,gcat in m.groupby('ranked_category'):
        for metric in METRICS:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for dim,gd in gcat.groupby('dimension'):
                gd=gd.sort_values('rank_number'); ax.plot(gd.rank_number, gd[metric], marker='o', label=f'H{int(dim)}')
                sub=gd[['rank_number',metric]].dropna(); rho,p=(spearmanr(sub.rank_number, sub[metric]) if len(sub)>=3 else (np.nan,np.nan)); rows.append({'ranked_category':cat,'dimension':dim,'metric':metric,'spearman_rho_rank_vs_metric':rho,'spearman_p':p})
            ax.set_title(f'{cat}: {metric} by rank'); ax.set_xlabel('Rank'); ax.set_ylabel(metric); ax.legend(); fig.tight_layout(); fig.savefig(plots/f'{safe_filename(cat)}_{metric}_by_rank.png', dpi=220); plt.close(fig)
    pd.DataFrame(rows).to_csv(tables/'rank_persistence_spearman_correlations.csv', index=False); print(f'Saved verification outputs to {out}')
if __name__=='__main__': main()
