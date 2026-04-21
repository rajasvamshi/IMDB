#!/usr/bin/env python3
"""
Low-memory subset LSTM trainer (for M1/M2 Macs).
Trains on a small slice of IMDb train set, saves model & tokenizer in models/,
and writes results/deep_eval.csv
"""
import os
import joblib
import argparse
import numpy as np
from data_loader import load_imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def build_model(vocab_size, maxlen=150, embed_dim=50):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
        Bidirectional(LSTM(32)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(data_dir='data/aclImdb', out_dir='models', subset=1000, maxlen=150, num_words=10000, epochs=2, batch_size=16):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading data...")
    train_df, test_df = load_imdb(data_dir)
    tr_sample = train_df.sample(n=min(subset, len(train_df)), random_state=42)
    X = tr_sample['review_text'].astype(str).tolist()
    y = tr_sample['sentiment'].astype(int).values

    print("Tokenizing...")
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(X)
    X_seq = tok.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=maxlen, padding='post', truncating='post')

    Xtr, Xval, ytr, yval = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
    vocab_size = min(num_words, len(tok.word_index) + 1)
    print(f"Vocab size: {vocab_size}, train samples: {len(Xtr)}")

    model = build_model(vocab_size=vocab_size, maxlen=maxlen, embed_dim=50)
    checkpoint = ModelCheckpoint(os.path.join(out_dir, 'lstm_final.h5'), save_best_only=True, monitor='val_loss')
    early = EarlyStopping(patience=2, monitor='val_loss', restore_best_weights=True)

    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early], verbose=1)

    joblib.dump(tok, os.path.join(out_dir, 'lstm_tokenizer.pkl'))

    # Evaluate on full test set
    print("Preparing test data...")
    test_texts = test_df['review_text'].astype(str).tolist()
    test_seq = tok.texts_to_sequences(test_texts)
    X_test = pad_sequences(test_seq, maxlen=maxlen, padding='post', truncating='post')
    y_test = test_df['sentiment'].astype(int).values
    preds = (model.predict(X_test, batch_size=256) > 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, preds)
    print(f"Deep model test accuracy: {acc:.4f}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'metric':['test_accuracy'],'value':[acc]}).to_csv('results/deep_eval.csv', index=False)
    print("Saved results/deep_eval.csv and model/tokenizer in models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/aclImdb')
    parser.add_argument('--out_dir', default='models')
    parser.add_argument('--subset', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--maxlen', type=int, default=150)
    parser.add_argument('--num_words', type=int, default=10000)
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, subset=args.subset, epochs=args.epochs, batch_size=args.batch_size, maxlen=args.maxlen, num_words=args.num_words)
