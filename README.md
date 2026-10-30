# English to French Machine Translation Comparison

**Yamir Singh Bamel · 220911280 · ICT4403 · Assignment 3**

## Execution status

<!-- STATUS_START -->
The implementation is prepared for a GPU notebook. Full model training and four-model
evaluation have **not yet been run**. No benchmark scores or best-performing model are
claimed in this initial package. `VALIDATION.md` records local checks separately.
The results section is populated automatically only after all four models finish.
<!-- STATUS_END -->

## Problem statement and objective

Translate English passages into French using a basic encoder–decoder Transformer,
fine-tuned mT5, prompted LLaMA and prompted Gemma. Compare BLEU, chrF, parameter count,
adaptation time and inference time, then examine the generated text for ethical risks.
The experiment compares practical systems under a small compute budget. It does not
isolate architecture from pretraining, model scale or adaptation differences.

## Dataset and preprocessing

Source: [Helsinki-NLP OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books),
configuration `en-fr`. This is a literary parallel corpus, not a representative sample
of all translation domains. Check the upstream dataset and underlying works' terms
before redistributing raw text; the repository ignores downloaded training data.

The pipeline normalizes Unicode to NFC, collapses whitespace, retains pairs with
2–60 whitespace-separated words per language and at most a 3:1 length ratio, and
removes repeated English **or** French strings using case-insensitive keys before
splitting. It shuffles with seed 42 and selects 10,000 training, 500 validation and
300 test pairs. The remaining eligible rows are unused. Filtering removes long or
unusual constructions and therefore narrows the evaluation population.

`results/dataset.json` records the downloaded revision, counts, average lengths and
split hashes. A joint byte-level BPE tokenizer of up to 8,000 tokens is trained only
on training pairs for the basic Transformer. Pretrained models retain their original
tokenizers. Encoder inputs and supervised targets are capped at 128 tokens.
Repeated exact strings cannot cross split boundaries, but related passages from the
same book may. Pretraining contamination cannot be excluded. This is a sentence-level
split, not a document-disjoint benchmark.

## Models

| Model | Architecture | Adaptation | Setup |
|---|---|---|---|
| Basic Transformer | Encoder–decoder | Train from scratch | 3 encoder and 3 decoder layers, width 256, 8 heads, feed-forward width 1024, dropout 0.1 |
| [mT5-small](https://huggingface.co/google/mt5-small) | Encoder–decoder | Full supervised fine-tuning | Adafactor, float32, gradient checkpointing |
| [LLaMA 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Decoder-only | Zero-shot prompting | Identical translation instruction, native chat template |
| [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) | Decoder-only | Zero-shot prompting | Identical translation instruction, native chat template |

Exact parameter counts are measured from loaded models rather than inferred from their
names. Model revisions are recorded. No quantization is applied. Decoder-only models
use bfloat16 where supported, otherwise float32. The basic Transformer and mT5 use
float32. These precision differences must be considered in computational comparisons.

Basic training uses AdamW, 10 epochs, batch size 32 and learning rate 0.0003. mT5 uses
3 epochs, microbatch size 2, accumulation 8 and learning rate 0.0005. Both use linear
warm-up/decay and gradient clipping. The best checkpoint is selected by validation
token loss, never by test scores. All settings are in `config.json`.

## Run in Google Colab

1. Upload `Assignment3_Colab.ipynb` at [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime. The notebook embeds the source package, so cloning is unnecessary.
3. Using your own Hugging Face account, obtain access to the linked official LLaMA and
   Gemma checkpoints. Enter a read token only in the notebook's private login prompt.
   Do not paste it in this README, commit it, or include it in screenshots.
4. Run the cells in order. The notebook checks model access before starting training.
5. Download the results ZIP at the end. It contains the updated README, source, measured
   scores, plots, generated translations and ethical-review worksheet. The checkpoint
   folder is excluded to keep the GitHub upload manageable.

Training can take substantial time; duration depends on the allocated GPU. Do not use
smoke-test results as assignment results. The notebook offers Drive storage for retaining
work across disconnections. Completed stages can be skipped on rerun. Interrupted training
stages must be explicitly restarted because optimizer state is not saved for mid-stage resume.

## Run locally

Use Python 3.10–3.12 and preferably a CUDA GPU. A CPU fallback supports small verification
runs, but the full experiment is designed for a GPU. Keep enough storage for model downloads
and checkpoints. Install a PyTorch build suitable for your GPU if the default is unsuitable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python -m pytest -q
python -m mt.prepare
python -m mt.train basic
python -m mt.train mt5
python -m mt.evaluate basic
python -m mt.evaluate mt5
python -m mt.evaluate llama
python -m mt.evaluate gemma
python -m mt.report
```

Commands must run from the repository root. Save `pip freeze` with each full run.
Do not change `config.json` midway through a run. The evaluation rejects checkpoints
whose configuration or training data no longer matches. To run a smaller experiment,
copy the project to a fresh run folder, edit the config **before preparation**, and label
that experiment accordingly. Seeds improve repeatability but GPU operations may still vary.

## Evaluation protocol

All models translate the same saved test examples in the same order. Main decoding is
greedy, beam count 1, maximum 128 new tokens and inference batch size 1. The basic
decoder masks future tokens and padding. mT5 receives the English-to-French task prefix;
LLaMA and Gemma receive the same zero-shot instruction with their own chat templates.
The test references are used for scoring only.

BLEU uses SacreBLEU's `13a` tokenizer; chrF uses character n-grams with `word_order=0`.
Both are corpus-level scores on a 0–100 scale, and signatures are saved. Each model is
warmed up using a validation source before timing. CUDA synchronization brackets the
timer. Inference time includes tokenization, generation and decoding, but excludes load,
download and scoring. Training time includes all epochs, validation and checkpoint writes,
but excludes preprocessing/download. Prompted models' task-training time is N/A, not a
claim that they required no pretraining. Hardware, dtype, output caps and source truncation
are recorded. Equal token budgets do not imply equal character budgets across tokenizers.

## Results obtained and conclusion

<!-- RESULTS_START -->
Full experiment pending. Run every model and `python -m mt.report` to populate the
measured comparison, temperature analysis, and a best-model conclusion based on BLEU.
No placeholder numerical scores are supplied.
<!-- RESULTS_END -->

## Sampling-temperature analysis

LLaMA and Gemma are additionally evaluated on the same first 60 fixed test examples at
temperatures 0.2, 0.7 and 1.0, using seeds 42, 43 and 44, top-p 0.9, and top-k disabled.
Greedy output is re-scored on that identical subset. The report gives mean and sample
standard deviation for BLEU across seeds, mean chrF and an error-bar plot. These trials
describe sampling behavior; they do not retune the main test comparison. Higher temperature
can introduce lexical diversity and factual errors, but the actual direction of score
changes is determined from measured outputs.

## Responsible and ethical AI analysis

Literary data may reproduce historical stereotypes and archaic or offensive usage.
Translation can change gender, remove uncertainty or negation, invent details, or replace
cultural concepts with superficially similar ones. Fluent text is not evidence of factual
faithfulness. BLEU and chrF cannot establish fairness or safety, and a single reference
may penalize valid paraphrases.

The ten hand-authored probes in `ethics_probes.json` examine explicit gender, ambiguity,
numbers, uncertainty, negation, idioms, cultural identity, respectful language and quoted
instructions. They are not used for training or headline metric computation. All four
models' actual outputs are saved separately. `results/ethics_review.csv` includes the
criterion for every probe and blank human verdict/explanation fields to complete after
inspection. Report examples of observed errors rather than asserting that every risk
actually occurred. The small probe set cannot certify safety or estimate group fairness.

Mitigations include better-balanced and carefully licensed training corpora, bilingual
human review of high-impact translations, explicit checks on numbers/names/negation,
preserving ambiguity when possible, culturally informed review, and labeling output as
machine-generated. Do not use test-set examples to improve prompts after seeing their
scores. Use an additional validation set if tuning the instruction. Keep sensitive text
out of public repositories; publish only appropriate samples under applicable terms.

## Repository contents

- `mt/`: preparation, Transformer implementation, training, inference and reporting.
- `config.json`, `requirements.txt`: experiment settings and dependencies.
- `tests/`: leakage, masking, padding, learning and metric checks.
- `results/`: measured outputs; initially only available preparation/validation evidence.
- `ethics_probes.json`: qualitative test cases and review criteria.
- `Assignment3_Colab.ipynb`: executable notebook with embedded source.

## Submission checklist

Complete the full run, review translations and ethical outputs, and commit the source,
results, figures and updated README to a GitHub repository. Do not upload caches, model
weights, tokens or the virtual environment. Give the evaluator access if private. Insert
the repository URL into the supplied Word sheet, save it as PDF and upload that PDF to
the designated LMS section. Confirm the deadline in the LMS; none is specified here.

## References

- [OPUS Books dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
- [mT5 model card](https://huggingface.co/google/mt5-small)
- [LLaMA model card and license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Gemma model card and license](https://huggingface.co/google/gemma-3-1b-it)
- [Gemma support in Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/model_doc/gemma3)
- [SacreBLEU implementation and metric signatures](https://github.com/mjpost/sacrebleu)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
