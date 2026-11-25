# concept-pipeline

Stages 2 to 6 of [Hebrew-English Concept Extraction](../README.md), as the Python package
`concept_pipeline`. It takes a `hebmath` extraction (stage 1, in `extractor/`) and returns the
concepts worth learning, ranked, each with the passages that define or use it.

| Module | Stage |
|---|---|
| `stage2_candidates.py`, `tokens.py`, `normalize.py`, `lexicon.py` | 2: candidate terms from seven sources |
| `stage2_model_normalizer.py` | 2: optional model base forms for Hebrew words |
| `stage4_merging.py` | 4: merge pairs, rule decisions, grouping |
| `stage4_translation_finder.py`, `stage4_synonym_finder.py`, `stage4_merge_decider.py` | 4: optional model finders and decider |
| `stage5_importance.py` | 5: importance score from eight signals, with calibrated weights |
| `stage6_selection.py` | 6: selection of the learning space's nodes |
| `llm_cache.py` | Record and replay of model calls, keyed by content |
| `gold.py`, `stage4_merge_eval.py`, `stage6_selection_eval.py` | Evaluation against gold labels |

No database and no network: model calls go through a sender function you pass in. The only
module that touches files is `llm_cache.py`, which keeps its folder of recorded answers. See
[docs/USAGE.md](../docs/USAGE.md) for installation, usage and the algorithmic and model modes.
