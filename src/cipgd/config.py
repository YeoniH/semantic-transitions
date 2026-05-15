from pathlib import Path

GLOBAL_DIALOGUES_REPO = Path('~/global-dialogues').expanduser()
GD_ROUND = 3
OUTPUT_DIR = Path(f'outputs/GD{GD_ROUND}')
MIN_RESPONSES = 30
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_POINTS_PER_QUESTION = 800
MAX_HOMOLOGY_DIMENSION = 2
PERSISTENCE_XLIM = (0.0, 0.82)
PERSISTENCE_YLIM = (0.0, 0.82)
RANDOM_SEED = 42
RANKED_CATEGORIES = [
    'highest_semantic_dispersion',
    'strongest_semantic_agreement_alignment',
    'highest_agreement_variability',
    'highest_mean_agreement',
    'lowest_mean_agreement',
    'lowest_semantic_dispersion',
    'candidate_turning_points',
    'candidate_bridge_questions',
    'candidate_consensus_frames',
]
