"""
Feature Set 2: Hand-crafted linguistic features for SST-2 sentiment classification.

Categories (per project plan):
  - Lexical:      positive/negative word counts, AFINN sentiment score
  - Syntax:       adjective / adverb / verb counts (spaCy POS tags)
  - Negation:     not / never / no / n't
  - Intensity:    very / really / extremely / absolutely / quite (+ small extra list)
  - Surface:      sentence length, avg word length, exclamation/question marks, ALL CAPS words
"""
import re
import numpy as np
import pandas as pd
import spacy
from afinn import Afinn
from tqdm import tqdm

NEGATION_WORDS = {"not", "never", "no", "n't", "nobody", "nothing", "neither",
                   "nowhere", "none", "cannot", "without", "hardly", "scarcely", "barely"}

INTENSIFIER_WORDS = {"very", "really", "extremely", "absolutely", "quite", "totally",
                      "completely", "utterly", "incredibly", "so", "too", "highly",
                      "remarkably", "particularly", "especially"}

FEATURE_NAMES = [
    "positive_words", "negative_words", "afinn_score",
    "adjective_count", "adverb_count", "verb_count", "noun_count",
    "negation_count",
    "intensifier_count",
    "sentence_length", "avg_word_length", "exclamation_marks",
    "question_marks", "all_caps_words", "comma_count",
]


class LinguisticFeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        self.af = Afinn()

    def _afinn_pos_neg_counts(self, tokens):
        pos, neg = 0, 0
        for t in tokens:
            s = self.af.score(t.lower())
            if s > 0:
                pos += 1
            elif s < 0:
                neg += 1
        return pos, neg

    def extract_one(self, doc, raw_sentence):
        tokens = [t.text for t in doc]
        n_tokens = max(len(tokens), 1)

        pos_words, neg_words = self._afinn_pos_neg_counts(tokens)
        afinn_score = self.af.score(raw_sentence)

        pos_tags = [t.pos_ for t in doc]
        adj_count = pos_tags.count("ADJ")
        adv_count = pos_tags.count("ADV")
        verb_count = pos_tags.count("VERB") + pos_tags.count("AUX")
        noun_count = pos_tags.count("NOUN") + pos_tags.count("PROPN")

        lower_tokens = [t.lower() for t in tokens]
        negation_count = sum(1 for t in lower_tokens if t in NEGATION_WORDS)
        intensifier_count = sum(1 for t in lower_tokens if t in INTENSIFIER_WORDS)

        word_lengths = [len(t) for t in tokens if t.isalpha()]
        avg_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0

        exclamation_marks = raw_sentence.count("!")
        question_marks = raw_sentence.count("?")
        comma_count = raw_sentence.count(",")
        all_caps_words = sum(1 for t in tokens if t.isalpha() and len(t) > 1 and t.isupper())

        return [
            pos_words, neg_words, afinn_score,
            adj_count, adv_count, verb_count, noun_count,
            negation_count,
            intensifier_count,
            n_tokens, avg_word_length, exclamation_marks,
            question_marks, all_caps_words, comma_count,
        ]

    def extract_batch(self, sentences, batch_size=256, n_process=1):
        rows = []
        docs = self.nlp.pipe(sentences, batch_size=batch_size, n_process=n_process)
        for doc, raw in tqdm(zip(docs, sentences), total=len(sentences), desc="linguistic features"):
            rows.append(self.extract_one(doc, raw))
        return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    ext = LinguisticFeatureExtractor()
    demo = [
        "This movie was absolutely wonderful and I loved every minute of it!",
        "This is not a good film, it was boring and a complete waste of time.",
    ]
    print(ext.extract_batch(demo))
