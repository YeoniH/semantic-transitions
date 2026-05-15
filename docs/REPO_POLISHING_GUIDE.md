# Repository polishing guide

Your working directory currently contains 14 numbered scripts (`00_inspect_embeddings.py` to `12_verify_turning_point_topology.py`), two utility modules, many README fragments, and a large `my_outputs/` tree. For a public repository, consolidate into one package module and fewer numbered scripts.

## Consolidation map

| Current file(s) | Polished version |
|---|---|
| `gd_utils.py`, `gd_stage2_utils.py` | `src/cipgd/utils.py` |
| hard-coded paths | `src/cipgd/config.py` |
| `00_inspect_embeddings.py`, `01_data_audit.py` | `scripts/00_data_audit.py` |
| `02_build_analysis_table.py` | `scripts/01_build_response_table.py` |
| `05_generate_response_embeddings.py` | `scripts/02_generate_response_embeddings.py` |
| `03_baseline_question_analysis.py`, `06_analyse_questions.py` | `scripts/03_question_semantic_analysis.py` |
| `10_sort_question_summary.py` | `scripts/04_rank_questions.py` |
| `04_optional_topology.py`, `07_topology_by_question.py`, `11a_*`, `11b_*` | `scripts/05_persistence_diagrams.py` |
| `12_verify_turning_point_topology.py` | `scripts/06_verify_turning_points.py` |
| `08_participant_trajectories.py` | `scripts/07_participant_trajectories.py` |
| `09_cross_group_comparison.py` | `scripts/08_cross_group_comparison.py` |

## Upload

Upload scripts, `src/`, docs, notebooks, requirements, and selected lightweight results. Exclude raw CIP data, response embeddings, parquet/pickle tables, and full generated directories.

## Best repo name

Best: `semantic-transitions-global-dialogues`.

Good: `cip-global-dialogues-analysis`.

Use caution: `cip-global-dialogues`, because it may look official.
