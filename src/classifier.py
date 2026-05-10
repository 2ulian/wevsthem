"""
Classifier utilities — Week 4 of "We vs Them" NLP project.

Provides:
  - build_tfidf_features(texts)        → sparse TF-IDF matrix
  - build_embedding_features(texts)    → dense sentence-transformer embeddings
  - train_and_evaluate(X, y, name)     → trains LR + SVM, returns results dict
  - save_model(model, path)            → pickle best model
  - load_model(path)                   → unpickle model
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_tfidf_features(
    texts: list[str],
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Fit a TF-IDF vectorizer and return (matrix, vectorizer).

    Parameters
    ----------
    texts        : list of strings to vectorize
    max_features : vocabulary cap
    ngram_range  : unigrams + bigrams by default

    Returns
    -------
    (X_sparse, vectorizer) — sparse scipy matrix + fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,      # use log(1+tf) instead of raw tf
        min_df=2,               # ignore terms that appear in fewer than 2 docs
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def build_embedding_features(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode texts using a sentence-transformers model.

    Parameters
    ----------
    texts        : list of strings
    model_name   : HuggingFace model identifier
    batch_size   : encoding batch size
    show_progress: show tqdm progress bar

    Returns
    -------
    np.ndarray of shape (n_texts, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train, X_test,
    y_train, y_test,
    feature_name: str,
    test_texts: list[str] | None = None,
) -> dict:
    """
    Train LogisticRegression and LinearSVC, evaluate both, return results.

    Parameters
    ----------
    X_train / X_test : feature matrices (sparse or dense)
    y_train / y_test : binary labels (0/1)
    feature_name     : label for printing (e.g. "TF-IDF")
    test_texts       : original texts for error analysis (optional)

    Returns
    -------
    dict with keys: best_model, best_name, results (list of per-model dicts)
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
        ),
    }

    results = []
    best_f1   = -1.0
    best_model = None
    best_name  = None

    for model_name, model in models.items():
        print(f"\n  Training {model_name} on {feature_name} features…")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=True
        )
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["no_othering", "othering"])

        print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(f"  Confusion matrix:\n{cm}")
        print(f"\n  Full report:\n{report}")

        # Error analysis — 10 wrong predictions
        if test_texts is not None:
            wrong_idx = np.where(y_pred != y_test)[0]
            print(f"  Wrong predictions: {len(wrong_idx)}")
            print(f"  --- 10 error examples ---")
            for i in wrong_idx[:10]:
                true_lbl  = "othering" if y_test[i] else "no_othering"
                pred_lbl  = "othering" if y_pred[i] else "no_othering"
                preview   = str(test_texts[i])[:120].replace("\n", " ")
                print(f"  true={true_lbl:<12} pred={pred_lbl:<12} | {preview}")

        entry = {
            "model_name": model_name,
            "feature":    feature_name,
            "model":      model,
            "precision":  prec,
            "recall":     rec,
            "f1":         f1,
            "cm":         cm,
            "report":     report,
        }
        results.append(entry)

        if f1 > best_f1:
            best_f1   = f1
            best_model = model
            best_name  = f"{model_name}_{feature_name}"

    return {"best_model": best_model, "best_name": best_name, "results": results}


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(obj: object, path: str) -> None:
    """Pickle an object (model + optional vectorizer) to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Model saved to {path}")


def load_model(path: str) -> object:
    """Load a pickled model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
