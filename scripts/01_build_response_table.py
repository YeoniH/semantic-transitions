from cipgd.config import GLOBAL_DIALOGUES_REPO, GD_ROUND, OUTPUT_DIR
from cipgd.utils import build_response_table, ensure_dir, save_table
import json

def main():
    out=ensure_dir(OUTPUT_DIR/'tables'); df=build_response_table(GLOBAL_DIALOGUES_REPO, GD_ROUND)
    path=save_table(df, out/'ask_opinion_response_table.parquet')
    df.head(5000).to_csv(out/'ask_opinion_response_table_preview.csv', index=False)
    (out/'analysis_table_metadata.json').write_text(json.dumps({'rows':len(df),'questions':df['question_id'].nunique(),'table':str(path)}, indent=2))
    print(f'Saved {path}')
if __name__=='__main__': main()
