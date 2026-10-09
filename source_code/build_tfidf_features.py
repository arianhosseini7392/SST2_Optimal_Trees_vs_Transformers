import os
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(name, col_map):
    df = pd.read_csv(f"{ROOT}/data/{name}.tsv", sep="\t")
    df = df.rename(columns=col_map)
    return df[["sentence", "label"]]


if __name__ == "__main__":
    train = load_split("train", {"sentence": "sentence"})
    dev = load_split("dev", {"content": "sentence"})
    test = load_split("test", {"content": "sentence"})

    train_clean = train["sentence"].map(clean)
    dev_clean = dev["sentence"].map(clean)
    test_clean = test["sentence"].map(clean)

    # Feature Set 1: TF-IDF, 5000-dim, fit on train only
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(train_clean)
    X_dev_tfidf = vectorizer.transform(dev_clean)
    X_test_tfidf = vectorizer.transform(test_clean)
    print("Full TF-IDF:", X_train_tfidf.shape)

    sp.save_npz(f"{ROOT}/data/train_tfidf.npz", X_train_tfidf)
    sp.save_npz(f"{ROOT}/data/dev_tfidf.npz", X_dev_tfidf)
    sp.save_npz(f"{ROOT}/data/test_tfidf.npz", X_test_tfidf)
    joblib.dump(vectorizer, f"{ROOT}/data/tfidf_vectorizer.joblib")

    # Reduced TF-IDF: top-200 terms by chi2, for use inside PySTREED (readable trees)
    K = 200
    selector = SelectKBest(chi2, k=K)
    selector.fit(X_train_tfidf, train["label"].values)
    selected_idx = selector.get_support(indices=True)
    vocab = np.array(vectorizer.get_feature_names_out())
    selected_terms = vocab[selected_idx]

    X_train_red = X_train_tfidf[:, selected_idx].toarray()
    X_dev_red = X_dev_tfidf[:, selected_idx].toarray()
    X_test_red = X_test_tfidf[:, selected_idx].toarray()

    pd.DataFrame(X_train_red, columns=[f"tfidf_{t}" for t in selected_terms]).to_csv(
        f"{ROOT}/data/train_tfidf_reduced.csv", index=False)
    pd.DataFrame(X_dev_red, columns=[f"tfidf_{t}" for t in selected_terms]).to_csv(
        f"{ROOT}/data/dev_tfidf_reduced.csv", index=False)
    pd.DataFrame(X_test_red, columns=[f"tfidf_{t}" for t in selected_terms]).to_csv(
        f"{ROOT}/data/test_tfidf_reduced.csv", index=False)

    print("Reduced TF-IDF:", X_train_red.shape)
    print("Top 20 selected terms by chi2:", list(selected_terms[:20]))
