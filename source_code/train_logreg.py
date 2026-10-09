import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_labels(name, col_map):
    df = pd.read_csv(f"{ROOT}/data/{name}.tsv", sep="\t").rename(columns=col_map)
    return df["label"].values


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    y_train = load_labels("train", {"sentence": "sentence"})
    y_dev = load_labels("dev", {"content": "sentence"})
    y_test = load_labels("test", {"content": "sentence"})

    X_train = sp.load_npz(f"{ROOT}/data/train_tfidf.npz")
    X_dev = sp.load_npz(f"{ROOT}/data/dev_tfidf.npz")
    X_test = sp.load_npz(f"{ROOT}/data/test_tfidf.npz")

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    pred_test = clf.predict(X_test)
    inference_time = (time.time() - t0) / len(y_test) * 1000  # ms/instance

    results = {
        "model": "LogisticRegression_TFIDF",
        "dev": evaluate(y_dev, clf.predict(X_dev)),
        "test": evaluate(y_test, pred_test),
        "train_time_sec": train_time,
        "inference_time_ms_per_instance": inference_time,
        "n_features": X_train.shape[1],
    }
    print(json.dumps(results, indent=2))

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/logreg_results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(clf, f"{ROOT}/results/logreg_model.joblib")
    np.save(f"{ROOT}/results/logreg_test_preds.npy", pred_test)
