# =============================================================================
# DistilBERT fine-tuning on SST-2 — run this in Google Colab (free GPU)
# Runtime > Change runtime type > T4 GPU, then run cells top to bottom.
#
# IMPORTANT: this uses the exact same train/dev/test TSV files used for the
# classical models (Logistic Regression + PySTREED), so the comparison in
# the paper is apples-to-apples. Upload sst2_data_for_colab.zip when prompted.
# =============================================================================

# %% Cell 1 — install packages (Colab already has torch; this adds the rest)
!pip install -q transformers datasets accelerate shap scikit-learn

# %% Cell 2 — upload the data zip (train.tsv / dev.tsv / test.tsv)
from google.colab import files
import zipfile, io

uploaded = files.upload()  # select sst2_data_for_colab.zip when prompted
zf = zipfile.ZipFile(io.BytesIO(uploaded['sst2_data_for_colab.zip']))
zf.extractall('data')
print(zf.namelist())

# %% Cell 3 — load data
import pandas as pd

train = pd.read_csv('data/train.tsv', sep='\t').rename(columns={'sentence': 'text'})
dev = pd.read_csv('data/dev.tsv', sep='\t').rename(columns={'content': 'text'})
test = pd.read_csv('data/test.tsv', sep='\t').rename(columns={'content': 'text'})

print('train:', train.shape, 'dev:', dev.shape, 'test:', test.shape)
print(train.head())

# %% Cell 4 — tokenize
from transformers import AutoTokenizer
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_hf_dataset(df):
    ds = Dataset.from_pandas(df[['text', 'label']])
    return ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=64),
                  batched=True)

train_ds = to_hf_dataset(train)
dev_ds = to_hf_dataset(dev)
test_ds = to_hf_dataset(test)

# %% Cell 5 — fine-tune
import numpy as np
import time
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
    }

# Note: deliberately not using per-epoch evaluation_strategy/eval_strategy here —
# the exact parameter name has changed across transformers versions. Dev and
# test are evaluated manually in Cell 6 instead, which is version-safe.
args = TrainingArguments(
    output_dir="distilbert_sst2",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
)

t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f"Training time: {train_time:.1f}s")

# %% Cell 6 — evaluate on dev and test, save predictions + probabilities
import torch
import json

def get_predictions(ds):
    t0 = time.time()
    preds_output = trainer.predict(ds)
    inference_time = (time.time() - t0) / len(ds) * 1000  # ms/instance
    logits = preds_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)
    return preds, probs, inference_time

dev_preds, dev_probs, _ = get_predictions(dev_ds)
test_preds, test_probs, test_inference_time = get_predictions(test_ds)

dev_metrics = {
    'accuracy': accuracy_score(dev['label'], dev_preds),
    'precision': precision_score(dev['label'], dev_preds),
    'recall': recall_score(dev['label'], dev_preds),
    'f1': f1_score(dev['label'], dev_preds),
}
test_metrics = {
    'accuracy': accuracy_score(test['label'], test_preds),
    'precision': precision_score(test['label'], test_preds),
    'recall': recall_score(test['label'], test_preds),
    'f1': f1_score(test['label'], test_preds),
}
print("DEV:", dev_metrics)
print("TEST:", test_metrics)

# save everything needed back in the main analysis
results_df = test.copy()
results_df['distilbert_pred'] = test_preds
results_df['distilbert_prob_positive'] = test_probs[:, 1]
results_df.to_csv('distilbert_test_predictions.csv', index=False)

summary = {
    'model': 'DistilBERT',
    'dev': dev_metrics,
    'test': test_metrics,
    'train_time_sec': train_time,
    'inference_time_ms_per_instance': test_inference_time,
    'n_params': sum(p.numel() for p in model.parameters()),
}
with open('distilbert_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

files.download('distilbert_test_predictions.csv')
files.download('distilbert_results.json')

# %% Cell 7 — SHAP explanations for a handful of example sentences
# (for the interpretability comparison section — local explanations vs. tree rules)
import shap

pred_pipeline = lambda texts: torch.softmax(
    torch.tensor(trainer.predict(to_hf_dataset(pd.DataFrame({'text': list(texts), 'label': [0]*len(texts)}))).predictions),
    dim=1
).numpy()

explainer = shap.Explainer(pred_pipeline, tokenizer)

example_sentences = [
    "This movie was absolutely wonderful and I loved every minute of it.",
    "It's so bad that it's actually entertaining.",         # sarcasm-ish, good for error analysis
    "This is not a good film, it was boring and dull.",     # negation
    "A stunning, beautifully acted, deeply moving drama.",
]
shap_values = explainer(example_sentences)

# save a simple text dump of top contributing tokens per example (for the paper)
with open('shap_examples.txt', 'w') as f:
    for i, sent in enumerate(example_sentences):
        f.write(f"Sentence: {sent}\n")
        tokens = shap_values.data[i]
        vals = shap_values.values[i][:, 1]  # contribution toward "positive" class
        pairs = sorted(zip(tokens, vals), key=lambda x: -abs(x[1]))[:8]
        for tok, val in pairs:
            f.write(f"   {tok!r}: {val:+.3f}\n")
        f.write("\n")

files.download('shap_examples.txt')

print("Done. Bring distilbert_test_predictions.csv, distilbert_results.json, "
      "and shap_examples.txt back to the main analysis.")
