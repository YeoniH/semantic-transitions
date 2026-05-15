from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

CORE_FILES = {
    'aggregate_standardized': 'GD{gd}_aggregate_standardized.csv',
    'binary': 'GD{gd}_binary.csv',
    'participants': 'GD{gd}_participants.csv',
    'preference': 'GD{gd}_preference.csv',
    'verbatim_map': 'GD{gd}_verbatim_map.csv',
    'summary': 'GD{gd}_summary.csv',
    'discussion_guide': 'GD{gd}_discussion_guide.csv',
    'embeddings': 'GD{gd}_embeddings.json',
}
CATEGORY_ABBREVIATIONS = {
    'highest_semantic_dispersion': 'HSD',
    'strongest_semantic_agreement_alignment': 'SSAA',
    'highest_agreement_variability': 'HAV',
    'highest_mean_agreement': 'HMA',
    'lowest_mean_agreement': 'LMA',
    'lowest_semantic_dispersion': 'LSD',
    'candidate_turning_points': 'CTP',
    'candidate_bridge_questions': 'CBQ',
    'candidate_consensus_frames': 'CCF',
}
def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve(); p.mkdir(parents=True, exist_ok=True); return p
def gd_dir(repo, gd): return Path(repo).expanduser().resolve() / 'Data' / f'GD{gd}'
def expected_paths(repo, gd):
    d = gd_dir(repo, gd); return {k: d / v.format(gd=gd) for k,v in CORE_FILES.items()}
def clean_text(x: Any) -> str:
    if pd.isna(x): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()
def canonical_text_key(x: Any) -> str:
    return clean_text(x).lower()
def safe_filename(s: str, max_len: int = 100) -> str:
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_')
    return s[:max_len] or 'output'
def coerce_numeric(s): return pd.to_numeric(s.astype(str).str.replace('%','',regex=False), errors='coerce')
def is_ask_opinion(s): return s.astype(str).str.contains('ask opinion', case=False, na=False)
def find_first_existing(df, candidates: Iterable[str]):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None
def load_aggregate(repo, gd):
    p = expected_paths(repo, gd)['aggregate_standardized']
    if not p.exists(): raise FileNotFoundError(f'Missing {p}')
    df = pd.read_csv(p, low_memory=False)
    if 'All' in df.columns: df['All'] = coerce_numeric(df['All'])
    return df
def read_csv_if_exists(path): return pd.read_csv(path, low_memory=False) if Path(path).exists() else None
def build_response_table(repo, gd):
    agg = load_aggregate(repo, gd)
    df = agg[is_ask_opinion(agg['Question Type'])].copy().reset_index(drop=True)
    df['response_text'] = df['Response'].map(clean_text)
    df['original_response_text'] = df.get('OriginalResponse', pd.Series(['']*len(df))).map(clean_text)
    df['question_text'] = df['Question'].map(clean_text)
    df['question_id'] = df['Question ID'].astype(str)
    df['participant_id'] = df.get('Participant ID', pd.Series([pd.NA]*len(df))).astype('string')
    df['language'] = df.get('Language', pd.Series([pd.NA]*len(df))).astype('string')
    df['agreement_all'] = df['All'] if 'All' in df.columns else np.nan
    df['response_row_id'] = np.arange(len(df))
    return df
def save_table(df, path):
    path = Path(path); ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False); return path
    except Exception:
        fallback = path.with_suffix('.pkl'); df.to_pickle(fallback); return fallback
def load_response_table(path):
    path = Path(path).expanduser()
    if path.suffix == '.parquet': return pd.read_parquet(path)
    if path.suffix in ['.pkl','.pickle']: return pd.read_pickle(path)
    return pd.read_csv(path, low_memory=False)
def load_embeddings(emb_path, idx_path):
    X = np.load(Path(emb_path).expanduser()); idx = pd.read_csv(Path(idx_path).expanduser(), low_memory=False)
    if len(X) != len(idx): raise ValueError('Embedding matrix and index length mismatch')
    return X, idx
def align_table_and_embeddings(df, X, idx):
    a = df.copy(); b = idx.copy(); a['response_row_id']=a['response_row_id'].astype(int); b['response_row_id']=b['response_row_id'].astype(int); b['_pos']=np.arange(len(b))
    m = a.merge(b[['response_row_id','_pos']], on='response_row_id', how='inner', validate='one_to_one')
    return m.drop(columns=['_pos']).reset_index(drop=True), X[m['_pos'].to_numpy()]
def numeric_agreement(df):
    return pd.to_numeric(df['agreement_all'], errors='coerce') if 'agreement_all' in df.columns else pd.Series(np.nan, index=df.index)
def mean_pairwise_cosine_distance(X, max_points=1500, seed=42):
    if len(X) < 2: return float('nan')
    if len(X) > max_points:
        rng=np.random.default_rng(seed); X=X[rng.choice(len(X), max_points, replace=False)]
    D=pairwise_distances(X, metric='cosine'); iu=np.triu_indices_from(D, k=1); return float(np.nanmean(D[iu]))
def semantic_centrality(X): return cosine_similarity(X).mean(axis=1) if len(X) else np.array([])
def choose_top_questions(df, min_responses=30, top_n=None):
    s=df.groupby('question_id')['response_row_id'].count().sort_values(ascending=False); s=s[s>=min_responses]
    if top_n: s=s.head(top_n)
    return [str(x) for x in s.index]
def add_rank_features(df):
    out=df.copy()
    for col in ['semantic_dispersion_mean_pairwise_cosine','agreement_mean','agreement_sd','pca_top_abs_agreement_rho']:
        if col in out.columns: out[f'{col}_pct_rank']=out[col].rank(pct=True)
    out['candidate_turning_point_score']=(out.get('semantic_dispersion_mean_pairwise_cosine_pct_rank',0).fillna(0)+out.get('pca_top_abs_agreement_rho_pct_rank',0).fillna(0)+out.get('agreement_sd_pct_rank',0).fillna(0))/3
    out['candidate_bridge_question_score']=(out.get('agreement_sd_pct_rank',0).fillna(0)+out.get('pca_top_abs_agreement_rho_pct_rank',0).fillna(0))/2
    out['candidate_consensus_frame_score']=(out.get('agreement_mean_pct_rank',0).fillna(0)+(1-out.get('semantic_dispersion_mean_pairwise_cosine_pct_rank',1).fillna(1)))/2
    return out
def finite_lifetimes(dgm):
    if len(dgm)==0: return np.array([])
    finite=dgm[np.isfinite(dgm[:,1])]
    return finite[:,1]-finite[:,0] if len(finite) else np.array([])
def diagram_stats(dgm):
    life=finite_lifetimes(dgm)
    return {'n_features':int(len(dgm)), 'n_finite_features':int(len(life)), 'mean_finite_lifetime':float(np.mean(life)) if len(life) else np.nan, 'max_finite_lifetime':float(np.max(life)) if len(life) else np.nan}
def draw_pd(ax, dgm, dim, xlim=(0,0.8), ylim=(0,0.8), show_metadata=True):
    stats=diagram_stats(dgm); finite=dgm[np.isfinite(dgm[:,1])] if len(dgm) else np.empty((0,2))
    if len(finite): ax.scatter(finite[:,0], finite[:,1], s=14, alpha=.75)
    ax.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]], '--', lw=1); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal', adjustable='box')
    if show_metadata:
        ax.text(.03,.97, f"features={stats['n_features']}\nfinite={stats['n_finite_features']}\nmean={stats['mean_finite_lifetime']:.3f}\nmax={stats['max_finite_lifetime']:.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox={'boxstyle':'round,pad=.25','alpha':.18,'lw':.5})
    return stats
def available_group_columns(df):
    return [c for c in ['language','Language','Country','country','Region','region','Gender','gender','Age','age','Education','education'] if c in df.columns]
