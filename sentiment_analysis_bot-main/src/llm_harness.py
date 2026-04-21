#!/usr/bin/env python3
"""
LLM harness for running prompt-based sentiment classification on IMDb test subset.
Writes results to CSV with columns:
 idx,true_label,pred_label,latency,raw_output
Uses modern OpenAI python client (openai>=1.x).
"""
import argparse
import os
import csv
import time
import random
from data_loader import load_imdb

# OpenAI modern client
from openai import OpenAI
import backoff

# Helper prompt templates
def zero_shot_prompt(review):
    return f"""Classify the sentiment of the following movie review as POSITIVE or NEGATIVE.
IMPORTANT: Reply with exactly one word ONLY — either POSITIVE or NEGATIVE. Do NOT add any explanation.

Review:
\"\"\"{review}\"\"\"

Answer:
"""

def few_shot_prompt(review, examples=3):
    exs = [
        ('I loved this movie; it had great acting and a touching story.', 'POSITIVE'),
        ('This movie was boring and too long.', 'NEGATIVE'),
        ('Stellar visuals and a brilliant plot. Highly recommended.', 'POSITIVE'),
        ('Waste of time. Poor direction and terrible acting.', 'NEGATIVE')
    ]
    pick = exs[:examples]
    s = ""
    for i,(r,l) in enumerate(pick,1):
        s += f"Example {i}:\nReview: \"{r}\"\nLabel: {l}\n\n"
    s += f"Now classify:\nReview: \"{review}\"\nLabel:"
    return s


def cot_prompt(review):
    return f"""Read the review and think step-by-step whether the sentiment is positive or negative. Then give the final label as POSITIVE or NEGATIVE.

Review:
\"\"\"{review}\"\"\"

Step-by-step reasoning:"""

def extract_label_from_text(out_text):
    """
    Heuristic to extract label from model text output.
    Returns: (label_int or None, raw_text)
    """
    if out_text is None:
        return None, ""
    text = out_text.strip().upper()
    if "POSITIVE" in text:
        return 1, out_text
    if "NEGATIVE" in text:
        return 0, out_text
    # fallback heuristics
    if any(w in text for w in ["GOOD", "GREAT", "LOVE", "FANTASTIC", "ENJOYED"]):
        return 1, out_text
    if any(w in text for w in ["BAD", "TERRIBLE", "BORING", "WORST", "HATED"]):
        return 0, out_text
    return None, out_text

# Backoff retry for transient errors (rate limits)
@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
def call_llm_with_retries(client, prompt, model="gpt-4o-mini", max_output_tokens=64, temperature=0.0):
    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
    # try output_text first (convenience property)
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # fallback attempt to parse structured output
    try:
        out = resp.output
        parts = []
        for item in out:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
                    elif isinstance(c, dict) and "text" in c:
                        parts.append(c["text"])
        return " ".join(parts).strip() if parts else ""
    except Exception:
        # last resort: stringify resp
        return str(resp)

def run_llm_on_subset(data_dir='data/aclImdb', subset=500, method='zero_shot', out_csv='results/llm_eval.csv', model='gpt-4o-mini', seed=42):
    # verify api key and client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    # load dataset
    _, test_df = load_imdb(data_dir)
    n = min(subset, len(test_df))
    test_df = test_df.sample(n=n, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    rows = []
    print(f"Running {n} samples with method={method} model={model} ...")
    for idx, row in test_df.iterrows():
        review = row['review_text']
        true_label = int(row['sentiment'])
        if method == 'zero_shot':
            prompt = zero_shot_prompt(review)
        elif method == 'few_shot':
            prompt = few_shot_prompt(review, examples=3)
        elif method == 'cot':
            prompt = cot_prompt(review)
        else:
            prompt = zero_shot_prompt(review)

        t0 = time.time()
        try:
            out_text = call_llm_with_retries(client, prompt, model=model, max_output_tokens=128, temperature=0.0)
            latency = time.time() - t0
            pred_label, raw = extract_label_from_text(out_text)
            # If label couldn't be extracted, set pred_label to empty and keep raw output
            rows.append({
                'idx': int(idx),
                'true_label': int(true_label),
                'pred_label': '' if pred_label is None else int(pred_label),
                'latency': float(latency),
                'raw_output': raw.replace("\n"," ").replace("\r"," ")
            })
            # brief progress print
            if (idx+1) % 10 == 0 or idx == 0:
                print(f"  processed {idx+1}/{n}  pred={rows[-1]['pred_label']} lat={rows[-1]['latency']:.2f}s")
        except Exception as e:
            latency = time.time() - t0
            rows.append({
                'idx': int(idx),
                'true_label': int(true_label),
                'pred_label': '',
                'latency': float(latency),
                'raw_output': f"ERROR: {e}"
            })
            print(f"  ERROR on idx={idx}: {e}")

    # write CSV
    fieldnames = ['idx','true_label','pred_label','latency','raw_output']
    with open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/aclImdb', help='Path to aclImdb folder')
    parser.add_argument('--subset', type=int, default=500, help='Number of test samples to run')
    parser.add_argument('--method', choices=['zero_shot','few_shot','cot'], default='zero_shot', help='Prompting method')
    parser.add_argument('--out', dest='out_csv', default='results/llm_eval.csv', help='CSV output path')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to call (must be accessible on your account)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_llm_on_subset(data_dir=args.data_dir, subset=args.subset, method=args.method, out_csv=args.out_csv, model=args.model, seed=args.seed)
