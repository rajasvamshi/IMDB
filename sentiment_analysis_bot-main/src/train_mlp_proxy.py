import os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_imdb

def main(data_dir='data/aclImdb', out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading data...")
    train_df, test_df = load_imdb(data_dir)
    X_train, y_train = train_df['review_text'], train_df['sentiment']
    X_test, y_test = test_df['review_text'], test_df['sentiment']

    print("Fitting TF-IDF...")
    vec = TfidfVectorizer(max_features=20000, stop_words='english')
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    print("Training small MLP (proxy for a deep model)...")
    mlp = MLPClassifier(hidden_layer_sizes=(512,256), max_iter=10, batch_size=128, learning_rate_init=0.001)
    mlp.fit(Xtr, y_train)

    print("Evaluating...")
    preds = mlp.predict(Xte)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)

    joblib.dump(mlp, os.path.join(out_dir, 'mlp_proxy.pkl'))
    joblib.dump(vec, os.path.join(out_dir, 'tfidf_vec_for_mlp.pkl'))
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'metric':['test_accuracy'],'value':[acc]}).to_csv('results/deep_eval.csv', index=False)
    print("Saved proxy model and results/deep_eval.csv")

if __name__ == '__main__':
    main()
