import os
import io
import json
import time
import contextlib
import pandas as pd
import numpy as np
from pystreed import STreeDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = f"{ROOT}/results"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(f"{RESULTS}/trees", exist_ok=True)


def load_labels(name, col_map):
    df = pd.read_csv(f"{ROOT}/data/{name}.tsv", sep="\t").rename(columns=col_map)
    return df["label"].values.astype(int)


def load_features(split, feature_set):
    if feature_set == "linguistic":
        df = pd.read_csv(f"{ROOT}/data/{split}_linguistic.csv")
        return df.drop(columns=["sentence", "label", "all_caps_words"])
    elif feature_set == "tfidf_reduced":
        return pd.read_csv(f"{ROOT}/data/{split}_tfidf_reduced.csv")
    elif feature_set == "hybrid":
        return pd.read_csv(f"{ROOT}/data/{split}_hybrid.csv")
    else:
        raise ValueError(feature_set)


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def capture_tree_text(clf):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        clf.print_tree()
    return buf.getvalue()


def tree_complexity(tree_text):
    lines = [l for l in tree_text.splitlines() if l.strip()]
    leaf_lines = [l for l in lines if "Label:" in l]
    n_leaves = len(leaf_lines)
    depths = []
    for l in leaf_lines:
        indent = len(l) - len(l.lstrip("| "))
        depth = l.count("|")
        depths.append(depth)
    avg_rule_length = float(np.mean(depths)) if depths else 0.0
    n_rules = n_leaves
    return n_leaves, avg_rule_length, n_rules


if __name__ == "__main__":
    y_train = load_labels("train", {"sentence": "sentence"})
    y_dev = load_labels("dev", {"content": "sentence"})
    y_test = load_labels("test", {"content": "sentence"})

    feature_sets = ["linguistic", "tfidf_reduced", "hybrid"]
    depths = [2, 3, 4]

    all_results = []

    for fs in feature_sets:
        X_train = load_features("train", fs)
        X_dev = load_features("dev", fs)
        X_test = load_features("test", fs)

        for depth in depths:
            print(f"=== {fs} | depth={depth} ===", flush=True)
            t0 = time.time()
            clf = STreeDClassifier(max_depth=depth, time_limit=200, verbose=False)
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            pred_test = clf.predict(X_test)
            inference_time = (time.time() - t0) / len(y_test) * 1000

            pred_dev = clf.predict(X_dev)

            tree_text = capture_tree_text(clf)
            n_leaves, avg_rule_length, n_rules = tree_complexity(tree_text)

            with open(f"{RESULTS}/trees/{fs}_depth{depth}.txt", "w") as f:
                f.write(tree_text)

            result = {
                "feature_set": fs,
                "depth": int(depth),
                "actual_depth": int(clf.get_depth()),
                "n_leaves": int(n_leaves),
                "n_rules": int(n_rules),
                "avg_rule_length": float(avg_rule_length),
                "dev": {k: float(v) for k, v in evaluate(y_dev, pred_dev).items()},
                "test": {k: float(v) for k, v in evaluate(y_test, pred_test).items()},
                "train_time_sec": float(train_time),
                "inference_time_ms_per_instance": float(inference_time),
                "n_features_input": int(X_train.shape[1]),
            }
            print(json.dumps(result, indent=2), flush=True)
            all_results.append(result)

            # cache predictions for error analysis later
            np.save(f"{RESULTS}/trees/{fs}_depth{depth}_test_preds.npy", pred_test)
            np.save(f"{RESULTS}/trees/{fs}_depth{depth}_dev_preds.npy", pred_dev)

            # write aggregate results incrementally so partial progress is never lost
            with open(f"{RESULTS}/pystreed_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    with open(f"{RESULTS}/pystreed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # flat CSV summary for quick viewing
    rows = []
    for r in all_results:
        rows.append({
            "feature_set": r["feature_set"], "depth": r["depth"],
            "n_leaves": r["n_leaves"], "avg_rule_length": r["avg_rule_length"],
            "dev_acc": r["dev"]["accuracy"], "dev_f1": r["dev"]["f1"],
            "test_acc": r["test"]["accuracy"], "test_f1": r["test"]["f1"],
            "train_time_sec": r["train_time_sec"],
        })
    pd.DataFrame(rows).to_csv(f"{RESULTS}/pystreed_summary.csv", index=False)
    print("\nDONE. Summary:")
    print(pd.DataFrame(rows).to_string(index=False))
