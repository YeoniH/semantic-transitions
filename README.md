# Semantic Transitions in Public Deliberation on AI

A preliminary computational workflow for analysing the Collective Intelligence Project (CIP)'s public **Global Dialogues** data.

This repository supports a research-fellowship proposal tentatively titled:

> **Semantic Transitions in Public Deliberation on AI**

The workflow explores how public articulates, organises, and collectively evaluates AI-related values in open-ended Global Dialogues responses. It combines response embeddings, semantic dispersion, PCA/UMAP, agreement metadata, ranked prompt categories, and exploratory topological analysis.

## Repository status

This is an **independent research workflow** using CIP's publicly available Global Dialogues data (https://github.com/collect-intel/global-dialogues). It is not an official CIP repository.

## Scope and evidential boundary

This workflow is designed around the currently available open Global Dialogues data. The open dataset supports analysis of:

- prompt-level open-ended responses;
- aggregate agreement and voting patterns;
- pairwise comparisons, where available;
- response metadata and demographic segmentation, where available;
- semantic and topological structure of response spaces.

The open dataset does **not**, by itself, establish within-person reasoning shifts, causal deliberative effects, or observed conversational "turning points", unless richer process-level information is available, such as ordered exposure to other perspectives, timestamps, pre/post responses, or participant-level deliberation trajectories.

Accordingly, this repository does **not** claim to detect actual deliberative turning points. Instead, it identifies **question-level discourse configurations** that may be informative for studying potential deliberative reorganisation.

In this workflow, the term **candidate transition-sensitive prompt** is preferred over "turning point". These are prompts whose response spaces show combinations of:

1. high semantic dispersion;
2. strong alignment between semantic structure and collective agreement;
3. high variability in agreement across responses;
4. persistent topological separation or trade-off structure in embedding space.

These should be interpreted as candidates for further qualitative and process-level investigation, not as direct evidence that participants changed their reasoning.

## Data

This repository does **not** redistribute CIP's Global Dialogues data.

Clone/download the public data separately:

```bash
git clone https://github.com/collect-intel/global-dialogues.git
```

Expected local structure:

```text
/path/to/global-dialogues/
  Data/
    GD3/
      GD3_aggregate_standardized.csv
      GD3_verbatim_map.csv
      GD3_binary.csv
      GD3_preference.csv
      GD3_participants.csv
      GD3_discussion_guide.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

## Configuration

Edit:

```text
src/cipgd/config.py
```

Example:

```python
from pathlib import Path

GLOBAL_DIALOGUES_REPO = Path("~/global-dialogues").expanduser()
GD_ROUND = 3
OUTPUT_DIR = Path(f"outputs/GD{GD_ROUND}")
```

## Quickstart

Recommended order:

```bash
python scripts/00_data_audit.py
python scripts/01_build_response_table.py
python scripts/02_generate_response_embeddings.py
python scripts/03_question_semantic_analysis.py
python scripts/04_rank_questions.py
python scripts/05_persistence_diagrams.py
python scripts/06_verify_transition_sensitive_prompts.py
```

Optional extensions:

```bash
python scripts/07_participant_trajectories.py
python scripts/08_cross_group_comparison.py
```

## Workflow (Script map)

| Step | Script | Purpose |
|---|---|---|
| 0 | `00_data_audit.py` | Inspect available files, question types, response counts, languages, and embedding-file schema. |
| 1 | `01_build_response_table.py` | Build a response-level table for Ask Opinion questions. |
| 2 | `02_generate_response_embeddings.py` | Generate multilingual response embeddings from `response_text`. |
| 3 | `03_question_semantic_analysis.py` | Compute semantic dispersion, PCA/UMAP maps, agreement correlations, and bridge candidates. |
| 4 | `04_rank_questions.py` | Produce ranked question lists and composite categories such as candidate transition-sensitive prompts. |
| 5 | `05_persistence_diagrams.py` | Compute persistent homology and persistence diagrams for ranked prompt categories. |
| 6 | `06_verify_transition_sensitive_prompts.py` | Test whether candidate transition-sensitive prompts show higher semantic dispersion and longer H0/H1 persistence. |
| 7 | `07_participant_trajectories.py` | Optional participant-level semantic-shift analysis, only meaningful where participant ordering is available. |
| 8 | `08_cross_group_comparison.py` | Optional cross-language/cross-demographic comparison. |

## What each method contributes

### PCA

PCA identifies dominant linear semantic axes in response embeddings. It is useful for testing whether a major semantic contrast is associated with aggregate agreement.

### UMAP

UMAP visualises local neighbourhoods and cluster-like structure. It is useful for exploratory inspection of discourse regions.

### Persistent homology

[Persistent homology](https://en.wikipedia.org/wiki/Persistent_homology) summarises the multi-scale shape of the response space. In this workflow:

- H0 persistence may indicate durable separation among discourse regions;
- H1 persistence may indicate loop-like or trade-off geometry;
- H2 is treated as exploratory and interpreted cautiously.

Persistent homology does not prove deliberative change. It complements PCA and UMAP by asking whether response spaces have multi-scale structure that may be relevant for future process-level analysis.

## AI assistance disclosure

Parts of the computational workflow, repository restructuring, documentation, and exploratory scripting were developed iteratively with assistance from OpenAI's ChatGPT as a programming and research-support tool.

All analytical decisions, methodological framing, interpretation, and final code review are the responsibility of the repository author.

## Interpretation caveat

This is an exploratory workflow. It should be read as a pilot for mapping semantic structures in public reasoning about AI, not as a definitive measurement of deliberative change. Claims about reasoning shifts would require richer process-level data or a design that directly observes change over time.
