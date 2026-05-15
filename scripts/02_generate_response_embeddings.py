from cipgd.config import OUTPUT_DIR, EMBEDDING_MODEL
from cipgd.utils import load_response_table, ensure_dir
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    path=OUTPUT_DIR/'tables'/'ask_opinion_response_table.parquet'
    if not path.exists(): path=path.with_suffix('.pkl')
    df=load_response_table(path); out=ensure_dir(OUTPUT_DIR/'response_embeddings')
    X=SentenceTransformer(EMBEDDING_MODEL).encode(df['response_text'].fillna('').astype(str).tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(out/'response_embeddings.npy', X)
    cols=[c for c in ['response_row_id','question_id','question_text','response_text','language','participant_id','agreement_all'] if c in df.columns]
    df[cols].to_csv(out/'response_embedding_index.csv', index=False)
    print(f'Saved embeddings {X.shape}')
if __name__=='__main__': main()
