import os
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import load_imdb

def train_baseline(data_dir='data/aclImdb', out_dir='models', max_features=20000):
    print("📥 Loading IMDb dataset ...")
    train_df, test_df = load_imdb(data_dir)
    X_train, y_train = train_df['review_text'], train_df['sentiment']
    X_test, y_test = test_df['review_text'], test_df['sentiment']

    print("🔧 Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("🤖 Training Logistic Regression baseline ...")
    lr = LogisticRegression(max_iter=200, solver='liblinear')
    lr.fit(X_train_tfidf, y_train)

    print("✅ Evaluating ...")
    train_acc = accuracy_score(y_train, lr.predict(X_train_tfidf))
    test_acc = accuracy_score(y_test, lr.predict(X_test_tfidf))
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(out_dir, 'tfidf_lr.pkl'))
    joblib.dump(tfidf, os.path.join(out_dir, 'tfidf_vec.pkl'))

    pd.DataFrame({
        'metric': ['train_accuracy', 'test_accuracy'],
        'value': [train_acc, test_acc]
    }).to_csv(os.path.join(out_dir, 'baseline_eval.csv'), index=False)
    print(f"📁 Saved model & metrics to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/aclImdb")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--max_features", type=int, default=20000)
    args = parser.parse_args()
    train_baseline(args.data_dir, args.out_dir, args.max_features)
