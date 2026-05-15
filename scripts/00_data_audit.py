from cipgd.config import GLOBAL_DIALOGUES_REPO, GD_ROUND, OUTPUT_DIR
from cipgd.utils import expected_paths, ensure_dir, load_aggregate, is_ask_opinion
import pandas as pd

def main():
    out=ensure_dir(OUTPUT_DIR/'audit'); paths=expected_paths(GLOBAL_DIALOGUES_REPO, GD_ROUND)
    pd.DataFrame([{'file_key':k,'path':str(p),'exists':p.exists(),'size_mb':round(p.stat().st_size/1e6,2) if p.exists() else None} for k,p in paths.items()]).to_csv(out/'file_inventory.csv', index=False)
    agg=load_aggregate(GLOBAL_DIALOGUES_REPO, GD_ROUND)
    agg.groupby('Question Type', dropna=False).agg(rows=('Question ID','count'), questions=('Question ID','nunique')).reset_index().to_csv(out/'question_type_summary.csv', index=False)
    op=agg[is_ask_opinion(agg['Question Type'])].copy()
    op.groupby('Question ID').agg(question=('Question','first'), n_responses=('Response','count'), mean_agreement=('All','mean'), sd_agreement=('All','std')).reset_index().sort_values('n_responses', ascending=False).to_csv(out/'ask_opinion_question_summary.csv', index=False)
    print(f'Wrote audit outputs to {out}')
if __name__=='__main__': main()
