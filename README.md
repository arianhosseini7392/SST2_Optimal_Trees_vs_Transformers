# Optimal Decision Trees vs. Transformers on SST-2

**Can optimal decision trees achieve competitive sentiment classification while providing substantially better interpretability than transformer-based models?**

An empirical comparison of optimal decision trees ([STreeD](https://github.com/AlgTUDelft/pystreed) via PySTREED), logistic regression, and a fine-tuned DistilBERT on the Stanford Sentiment Treebank (SST-2).

> Short answer: **not on raw accuracy** (70.2% vs. 93.0%), but the trees are small, exact, global rule sets, which is something post-hoc explanations of a transformer can't offer.

**Author:** Arian Hosseini, University of Tehran

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Main Findings](#main-findings)
- [Models and Features](#models-and-features)
- [Example Learned Rules](#example-learned-rules)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Fine-tuned transformers routinely exceed 90% accuracy on sentiment benchmarks, but their decision process isn't directly inspectable. Practitioners must rely on post-hoc tools such as SHAP or LIME, which approximate model behavior locally. Shallow decision trees are the opposite: every prediction traces through a short sequence of readable conditions.

Classical trees (CART) are built greedily. **Optimal** decision trees instead use dynamic programming with bounds to find the tree that provably maximizes training accuracy for a given depth. This project asks how far that gets us on a real NLP task.

We compare three model families on the same SST-2 train/validation/test split along four axes:

1. **Accuracy** (accuracy, precision, recall, F1)
2. **Model complexity** (depth, number of rules, rule length, parameters)
3. **Efficiency** (training and per-instance inference time)
4. **Explanation type** and **linguistic error patterns**

## Key Results

Test-set performance (1,821 labeled sentences; majority baseline = 50.1%):

| Model | Features | Test Acc. | Precision | Recall | F1 | Complexity |
|---|---|---|---|---|---|---|
| Majority baseline | n/a | 50.1% | n/a | n/a | n/a | trivial |
| PySTREED | TF-IDF reduced (200) | 55.3% | 52.9% | 94.1% | 67.7% | 15 leaves, depth 4 |
| PySTREED | Linguistic (14) | 68.0% | 64.2% | 81.1% | 71.7% | 8 leaves, depth 3 |
| PySTREED | Hybrid (214) | 70.2% | 66.9% | 79.9% | 72.8% | 16 leaves, depth 4 |
| Logistic Regression | TF-IDF (5,000) | 81.1% | 79.0% | 84.5% | 81.7% | 5,000 weights |
| DistilBERT | raw text | **93.0%** | 91.3% | 94.9% | 93.1% | 66.96M params |

Training and inference cost (classical models on one CPU core; DistilBERT on a T4 GPU, so training times are not directly comparable):

| Model | Train time | Inference (ms/instance) |
|---|---|---|
| Logistic Regression | 0.42 s | 0.0002 |
| PySTREED (linguistic, depth 3) | 0.93 s | 0.0055 |
| PySTREED (hybrid, depth 4) | 120.8 s | 0.0396 |
| DistilBERT | 1143 s (19.0 min) | 1.71 |

## Main Findings

- **Trees trade accuracy for full transparency.** The best optimal tree (hybrid features, depth 4, 16 rules) reaches 70.2%, about 11 points below Logistic Regression and 23 points below DistilBERT.
- **Feature choice matters more than depth.** Aggregated linguistic features (e.g. a sentence-level AFINN score) use a shallow tree's depth budget far more efficiently than sparse TF-IDF terms. The TF-IDF-only tree stays close to the majority baseline at every depth.
- **Hybrid features win.** Combining 14 linguistic features with the 200 most label-associated TF-IDF terms gives a small, consistent edge, and unlike the linguistic-only tree it keeps improving through depth 4.
- **The transformer's advantage is concentrated.** Contrastive conjunctions (*but, though, however, yet, despite*) appear in 23.6% of sentences that only DistilBERT gets right, vs. 12.0% of sentences everyone gets right and 13.7% of the test set overall. None of the feature sets represent clause-level contrast directly.
- **Only PySTREED gives an explanation that is both global and exact.**

| Model | Explanation type | Scope | Exact or approximate |
|---|---|---|---|
| Logistic Regression | Feature weights | Global | Exact |
| PySTREED | Decision rules | Global | Exact |
| DistilBERT | SHAP attribution | Local (per instance) | Approximate |

## Models and Features

### Feature sets (classical models)

| Set | Description |
|---|---|
| **TF-IDF** | 5,000-dim unigram + bigram TF-IDF (min doc freq 2). Used for Logistic Regression. |
| **Linguistic** | 14 hand-crafted features: AFINN positive/negative counts and sentiment score; adjective/adverb/verb/noun counts (spaCy); negation count; intensifier count; sentence length, average word length, exclamation, question mark, and comma counts. |
| **Hybrid** | The 14 linguistic features plus the top 200 TF-IDF terms by chi-squared score (214 total). |

> An all-caps word count was dropped from the linguistic set: SST-2 is lowercased, so the feature was constant at zero.

### Models

- **Logistic Regression**: scikit-learn, L2 regularization, `C = 1.0`, on the full 5,000-dim TF-IDF vectors.
- **PySTREED**: optimal trees at depths 1 to 4 on each feature set (plus depth 5 for the linguistic set).
- **DistilBERT**: `distilbert-base-uncased` fine-tuned for 3 epochs (batch size 32, learning rate 2e-5, max length 64) on raw text.
- **SHAP**: used for per-instance explanations of DistilBERT.

## Example Learned Rules

The depth-3 linguistic tree (8 rules) is the cleanest example. It resolves negation first, then sentiment magnitude:

```text
IF negation_count > 0.5 AND sentence_length > 18 AND afinn_score > 1.0            -> Positive
IF negation_count > 0.5 AND sentence_length > 18 AND afinn_score <= 1.0           -> Negative
IF negation_count > 0.5 AND sentence_length <= 18 AND no exclamation mark         -> Negative
IF negation_count > 0.5 AND sentence_length <= 18 AND has exclamation mark        -> Positive
IF negation_count <= 0.5 AND intensifier_count > 0.5 AND afinn_score > 0.0        -> Positive
IF negation_count <= 0.5 AND intensifier_count > 0.5 AND afinn_score <= 0.0       -> Negative
IF negation_count <= 0.5 AND intensifier_count <= 0.5 AND afinn_score > -1.0      -> Positive
IF negation_count <= 0.5 AND intensifier_count <= 0.5 AND afinn_score <= -1.0     -> Negative
```

The depth-4 hybrid tree (16 rules) keeps the same top-level structure (negation, then AFINN score) but its deeper splits rely on individual terms such as `tfidf_film`, `tfidf_funny`, `tfidf_like`, and `tfidf_compelling`.

## Getting Started

### Prerequisites

- Python 3.9+
- A GPU is recommended for fine-tuning DistilBERT (the paper used a T4); everything else runs on CPU.

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Core dependencies: `scikit-learn`, `pystreed`, `transformers`, `datasets`, `torch`, `shap`, `spacy`, `afinn`, `pandas`, `numpy`, `matplotlib`.

## Project Structure

> Adjust this section to match your actual repository layout.

```text
.
├── data/                 # SST-2 splits (downloaded or cached here)
├── notebooks/            # Experiments and analysis
├── src/
│   ├── features.py       # TF-IDF, linguistic, and hybrid feature builders
│   ├── train_logreg.py   # Logistic Regression baseline
│   ├── train_streed.py   # PySTREED trees (all feature sets and depths)
│   ├── train_distilbert.py
│   └── analysis.py       # Error analysis, SHAP, rule extraction
├── results/              # Metrics, figures, extracted rules
├── paper/                # Written report (PDF)
├── requirements.txt
└── README.md
```

## Reproducing the Experiments

> Replace the commands below with your actual entry points.

```bash
# 1. Build features
python src/features.py

# 2. Baseline
python src/train_logreg.py

# 3. Optimal trees (depths 1-4; depth 5 for linguistic features)
python src/train_streed.py --features linguistic --max-depth 5
python src/train_streed.py --features tfidf_reduced --max-depth 4
python src/train_streed.py --features hybrid --max-depth 4

# 4. Transformer
python src/train_distilbert.py --epochs 3 --batch-size 32 --lr 2e-5

# 5. Error analysis and SHAP explanations
python src/analysis.py
```

**Data:** the official SST-2 train/validation/test split with gold labels on all three partitions: 67,349 train, 872 validation, and 1,821 test sentences. All models are trained and evaluated on identical splits.

**Compute note:** hybrid depth 4 took about two minutes on a single CPU core, and STreeD's search time grows super-linearly with depth.

## Limitations

- Single random seed, no cross-validation, and no confidence intervals, so small differences (e.g. hybrid depth 3 vs. 4) should be read cautiously.
- Tree depth was limited to 1 to 5 because of STreeD's search cost.
- Chi-squared filtering is a simple feature-selection method; a better one might change the weak TF-IDF-only result.
- SST-2 is short, single-clause, pre-normalized movie-review text. Findings may not transfer to longer or noisier text.
- DistilBERT was trained once with fixed hyperparameters and no search.
- The 14 linguistic features are a design choice, not an exhaustive account of sentiment structure.

**Future work:** add a feature for contrastive conjunctions and their position in the sentence, targeting the clearest gap between the interpretable models and DistilBERT.

## References

- Devlin et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL-HLT.
- Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS 30.
- Nielsen (2011). *A new ANEW: Evaluation of a word list for sentiment analysis in microblogs.* ESWC Workshop on Making Sense of Microposts.
- Pedregosa et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR 12.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Socher et al. (2013). *Recursive deep models for semantic compositionality over a sentiment treebank.* EMNLP.
- van der Linden, de Weerdt & Demirović (2023). *Necessary and sufficient conditions for optimal decision trees using dynamic programming.* NeurIPS 36.
- Wang et al. (2018). *GLUE: A multi-task benchmark and analysis platform for natural language understanding.* BlackboxNLP.

## License

Add your license here (e.g. MIT).

## Citation

If you use this work, please cite:

```bibtex
@misc{hosseini_sst2_optimal_trees,
  author = {Hosseini, Arian},
  title  = {Can Optimal Decision Trees Achieve Competitive Sentiment Classification While Providing Substantially Better Interpretability Than Transformer-Based Models?},
  note   = {University of Tehran}
}
```
