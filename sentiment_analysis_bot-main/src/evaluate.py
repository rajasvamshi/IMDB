import os
import argparse
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from data_loader import load_imdb


# ---------- Helper: detect text/label columns ----------

def detect_text_label_cols(df: pd.DataFrame):
    """
    Try to detect which columns hold the review text and the sentiment label.
    This makes the evaluator robust even if the column names change slightly.
    """
    text_col = None
    label_col = None

    # common guesses for text
    for cand in ["review", "review_text", "text", "content", "sentence"]:
        if cand in df.columns:
            text_col = cand
            break

    # common guesses for label
    for cand in ["sentiment", "label", "target", "y", "class"]:
        if cand in df.columns:
            label_col = cand
            break

    # fallbacks (try first / last columns if still None)
    if text_col is None and len(df.columns) > 0:
        text_col = df.columns[0]
        print(f"⚠️ Could not find standard text column; using first column: {text_col}")

    if label_col is None and len(df.columns) > 1:
        label_col = df.columns[-1]
        print(f"⚠️ Could not find standard label column; using last column: {label_col}")

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns. Available columns: {list(df.columns)}"
        )

    return text_col, label_col


# ---------- Baseline evaluation ----------

def eval_baseline(data_dir: str, models_dir: str, out_dir: str):
    print("🔹 Evaluating TF-IDF + Logistic Regression baseline...")

    vec_path = os.path.join(models_dir, "tfidf_vec.pkl")
    lr_path = os.path.join(models_dir, "tfidf_lr.pkl")

    if not (os.path.exists(vec_path) and os.path.exists(lr_path)):
        print("⚠️ Baseline model files not found, skipping baseline.")
        return

    tfidf_vec = joblib.load(vec_path)
    lr_model = joblib.load(lr_path)

    train_df, test_df = load_imdb(data_dir)
    text_col, label_col = detect_text_label_cols(test_df)

    X_test = tfidf_vec.transform(test_df[text_col].astype(str).tolist())
    y_test = test_df[label_col].astype(int).values

    y_pred = lr_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("✅ Baseline results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    df_out = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1", "value": f1},
        ]
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_eval.csv")
    df_out.to_csv(out_path, index=False)
    print(f"📁 Baseline metrics saved to: {out_path}")


# ---------- Deep (NB proxy) evaluation ----------

def eval_deep_nb(data_dir: str, models_dir: str, out_dir: str):
    """
    Evaluate the Naive Bayes 'deep proxy' model on the IMDb test set.
    Expects:
      - models/tfidf_vec_for_nb.pkl
      - models/nb_proxy.pkl
    """
    print("🔹 Evaluating Naive Bayes deep proxy model...")

    vec_path = os.path.join(models_dir, "tfidf_vec_for_nb.pkl")
    nb_path = os.path.join(models_dir, "nb_proxy.pkl")

    if not (os.path.exists(vec_path) and os.path.exists(nb_path)):
        print("⚠️ Deep NB proxy files not found, skipping deep model.")
        return

    tfidf_vec_nb = joblib.load(vec_path)
    nb_model = joblib.load(nb_path)

    train_df, test_df = load_imdb(data_dir)
    text_col, label_col = detect_text_label_cols(test_df)

    X_test = tfidf_vec_nb.transform(test_df[text_col].astype(str).tolist())
    y_test = test_df[label_col].astype(int).values

    y_pred = nb_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("✅ Deep NB proxy results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    df_out = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1", "value": f1},
        ]
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deep_eval.csv")
    df_out.to_csv(out_path, index=False)
    print(f"📁 Deep NB metrics saved to: {out_path}")


# ---------- LLM evaluation ----------

def eval_llm(llm_csv: str, out_dir: str):
    """
    Evaluate LLM predictions from a CSV file.
    Expects a 'clean' csv with columns:
      - true / true_label / ...
      - pred / pred_label / ...
    """
    print("🔹 Evaluating LLM results from:", llm_csv)

    if not llm_csv or not os.path.exists(llm_csv):
        print("⚠️ LLM CSV not found, skipping LLM evaluation.")
        return

    df = pd.read_csv(llm_csv)

    true_col = None
    pred_col = None
    for cand in ["true", "true_label", "label_true", "y_true"]:
        if cand in df.columns:
            true_col = cand
            break
    for cand in ["pred", "pred_label", "label_pred", "y_pred"]:
        if cand in df.columns:
            pred_col = cand
            break

    if true_col is None or pred_col is None:
        print("⚠️ Could not find true/pred columns in LLM CSV, skipping.")
        print("   Columns present:", list(df.columns))
        return

    df_valid = df.dropna(subset=[pred_col])
    y_true = df_valid[true_col].astype(int).values
    y_pred = df_valid[pred_col].astype(int).values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print(f"Total samples in CSV: {len(df)}")
    print(f"Valid (non-missing) predictions: {len(df_valid)}")
    print("✅ LLM results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    df_out = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1", "value": f1},
            {"metric": "num_samples", "value": len(df)},
            {"metric": "num_valid", "value": len(df_valid)},
        ]
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llm_eval_summary.csv")
    df_out.to_csv(out_path, index=False)
    print(f"📁 LLM metrics saved to: {out_path}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline, deep NB proxy, and LLM for IMDb sentiment."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/aclImdb",
        help="Path to IMDb dataset root.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing trained models.",
    )
    parser.add_argument(
        "--llm_csv",
        type=str,
        default="results/llm_eval_clean.csv",
        help="CSV with LLM predictions (cleaned).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for evaluation CSVs.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    eval_baseline(args.data_dir, args.models_dir, args.out_dir)
    eval_deep_nb(args.data_dir, args.models_dir, args.out_dir)
    eval_llm(args.llm_csv, args.out_dir)

    print("📊 All evaluation complete.")


if __name__ == "__main__":
    main()
