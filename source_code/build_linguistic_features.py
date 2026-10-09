import os
import pandas as pd
from linguistic_features import LinguisticFeatureExtractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_split(name, col_map):
    df = pd.read_csv(f"{ROOT}/data/{name}.tsv", sep="\t")
    df = df.rename(columns=col_map)
    return df[["sentence", "label"]]

if __name__ == "__main__":
    train = load_split("train", {"sentence": "sentence"})
    dev = load_split("dev", {"content": "sentence"})
    test = load_split("test", {"content": "sentence"})

    ext = LinguisticFeatureExtractor()

    for name, df in [("train", train), ("dev", dev), ("test", test)]:
        feats = ext.extract_batch(df["sentence"].tolist(), batch_size=256)
        out = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
        out.to_csv(f"{ROOT}/data/{name}_linguistic.csv", index=False)
        print(name, out.shape)
