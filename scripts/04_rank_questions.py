from cipgd.config import OUTPUT_DIR
from cipgd.utils import add_rank_features, ensure_dir
import pandas as pd
SORTS={'highest_semantic_dispersion':('semantic_dispersion_mean_pairwise_cosine',False),'strongest_semantic_agreement_alignment':('pca_top_abs_agreement_rho',False),'highest_agreement_variability':('agreement_sd',False),'highest_mean_agreement':('agreement_mean',False),'lowest_mean_agreement':('agreement_mean',True),'lowest_semantic_dispersion':('semantic_dispersion_mean_pairwise_cosine',True),'candidate_turning_points':('candidate_turning_point_score',False),'candidate_bridge_questions':('candidate_bridge_question_score',False),'candidate_consensus_frames':('candidate_consensus_frame_score',False)}
def main():
    out=ensure_dir(OUTPUT_DIR/'question_analysis'/'ranked_views'); df=add_rank_features(pd.read_csv(OUTPUT_DIR/'question_analysis'/'tables'/'all_question_summary.csv')); df.to_csv(out/'all_question_summary_enriched.csv', index=False)
    for name,(col,asc) in SORTS.items(): df.sort_values(col, ascending=asc, na_position='last').to_csv(out/f'{name}.csv', index=False)
    print(f'Saved ranked views to {out}')
if __name__=='__main__': main()
