# Bounded translation-only development probe

This is a separate exploratory branch of the research method, proposed **after** the original V2 final outputs were inspected. It does not modify, replace, or retune the frozen V2 candidate. The original final dataset cannot serve as independent evidence for selecting this new method.

The probe asks whether sentence translation by NLLB-200 distilled 600M (public CTranslate2 int8 conversion), plus the existing source-linked glossary, improves fidelity and speed on the existing development reports. It is not a new fine-tuned model and does not simplify Chinese input. It does not use a model critic or treat lexical checks as accuracy.

The official [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M) specifies CC-BY-NC-4.0, research use, general-domain sentence translation, and explicitly says it is not intended for medical or document translation. This bounded diagnostic investigates that limitation; it is not a production recommendation. The [CTranslate2 documentation](https://opennmt.net/CTranslate2/guides/transformers.html#nllb) describes the required source-language tokens and target-language prefix.

Only 14 English entries from the original 20 synthetic development cases and all 16 real development reports are used. The six Chinese inputs are excluded because this experiment is English-to-Chinese translation, not Chinese simplification. Parameters and selection criteria are recorded before inference in `probe_plan.json`.

No new inference overlaps the current frozen V2 worker. Environment and weights live under ignored `artifacts/v2`, separate from `.venv-mac` and all other projects. If the probe is not better, retain its output and reject it. If it is better, freeze a new independent test set and candidate before measuring a new final comparison; do not reuse already-inspected final cases as unseen evidence.
