# Validation record

Prepared on 30 October 2026 for Yamir Singh Bamel, registration 220911280.

## Checks completed locally

- Downloaded the real OPUS Books English–French dataset successfully: 127,085 raw pairs.
- Cleaning retained 119,415 eligible pairs. Selected 10,000 training, 500 validation and
  300 test pairs. Exact split hashes and dataset revision are in `results/dataset.json`.
- Six automated checks passed: normalization and duplicate handling; causal attention
  masking; source padding masking; tokenizer, optimization and generation; metric
  correctness/alignment; and tiny random mT5/LLaMA/Gemma model API compatibility.
- A separate small basic-Transformer experiment completed training, checkpoint saving,
  reloading, translation generation, metric computation and qualitative probe generation.
  It used 16 training pairs and a miniature model solely as an integration check. Its
  numerical scores are intentionally excluded from assignment results.
- Report tables, figures and README updates passed an integration check using isolated
  synthetic fixtures; those fixtures are not included in deliverables.
- Notebook code cells passed Python syntax compilation.
- The one-page Word submission sheet was converted to PDF and visually reviewed.

Test environment: macOS ARM, Python 3.12, PyTorch 2.8.0, Transformers 4.51.3,
Datasets 3.5.0, SacreBLEU 2.5.1. A PyTorch prototype nested-tensor warning was emitted;
it did not cause test failures. This does not establish successful execution in Colab.

## Work still required for final submission

Run the full experiment in a GPU runtime with authorized access to both gated model
checkpoints. Actual pretrained mT5/LLaMA/Gemma end-to-end runs have not been verified
locally. Review the resulting translations and ethical probes, publish the repository,
insert its real URL into the cover sheet, regenerate the PDF and submit it to the LMS.
No fabricated model measurements or invented repository URL are included.
