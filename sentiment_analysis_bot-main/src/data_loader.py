import os
import pandas as pd

def load_imdb(data_dir):
    """
    Load IMDb dataset from the given directory.
    Expected folder structure:
    data_dir/
      train/pos, train/neg, test/pos, test/neg
    Returns:
      train_df, test_df  (each a pandas DataFrame with columns ['review_text', 'sentiment'])
    """
    def read_data(split):
        reviews = []
        for label in ['pos', 'neg']:
            path = os.path.join(data_dir, split, label)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")
            for fname in os.listdir(path):
                if fname.endswith(".txt"):
                    with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        sentiment = 1 if label == 'pos' else 0
                        reviews.append((text, sentiment))
        df = pd.DataFrame(reviews, columns=["review_text", "sentiment"])
        return df

    train_df = read_data("train")
    test_df = read_data("test")
    return train_df, test_df

