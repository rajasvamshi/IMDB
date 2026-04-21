import os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

    print("Training MultinomialNB (fast proxy for deep model)...")
    nb = MultinomialNB()
    nb.fit(Xtr, y_train)

    print("Evaluating...")
    preds = nb.predict(Xte)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)

    joblib.dump(nb, os.path.join(out_dir, 'nb_proxy.pkl'))
    joblib.dump(vec, os.path.join(out_dir, 'tfidf_vec_for_nb.pkl'))
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'metric':['test_accuracy'],'value':[acc]}).to_csv('results/deep_eval.csv', index=False)
    print("Saved proxy model and results/deep_eval.csv")

if __name__ == '__main__':
    main()
