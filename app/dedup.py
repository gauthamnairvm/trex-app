import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def trex_dedup(df, text_col, stopword_flag=True):
    os.makedirs("results/dedup", exist_ok=True)

    print("[+] Starting deduplication pipeline...")

    # Step 1: Text Cleaning
    print("[?] Remove stopwords? (y/n): ", end="")
    if stopword_flag:
        use_default = input("Use default stopwords? (y/n): ").strip().lower() == 'y'
        if use_default:
            stopword_set = set(stopwords.words('english'))
        else:
            custom_sw = input("Enter custom stopwords (comma-separated): ").split(',')
            stopword_set = set(word.strip().lower() for word in custom_sw)
    else:
        stopword_set = set()

    def clean_text(text):
        tokens = word_tokenize(text)
        return " ".join([t for t in tokens if t.lower() not in stopword_set])

    df['clean_text'] = df[text_col].astype(str).apply(clean_text)

    # Step 2: MinHash LSH Deduplication
    print("[+] Running MinHash LSH...")
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    minhashes = {}

    for i, row in df.iterrows():
        tokens = set(row['clean_text'].split())
        m = MinHash(num_perm=128)
        for t in tokens:
            m.update(t.encode('utf8'))
        lsh.insert(str(row['row_id']), m)
        minhashes[row['row_id']] = m

    duplicate_types = []
    for i, row in df.iterrows():
        row_id = str(row['row_id'])
        result = lsh.query(minhashes[row['row_id']])
        result = set(result) - {row_id}
        if not result:
            duplicate_types.append('unique')
        else:
            max_sim = max(minhashes[row['row_id']].jaccard(minhashes[int(r)]) for r in result)
            if max_sim == 1.0:
                duplicate_types.append('exact_duplicate')
            else:
                duplicate_types.append('near_duplicate')

    df['dup_type'] = duplicate_types

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='dup_type')
    plt.title("Duplicate Type Distribution")
    plt.tight_layout()
    plt.savefig("results/dedup/dup_type_distribution.png")
    print("[+] Saved dup_type_distribution.png")

    # Step 3: 3-Gram + Entropy
    print("[+] Computing Shannon Entropy and 3-gram overlap...")
    def shannon_entropy(text):
        prob = [c / len(text) for c in Counter(text).values()]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    df['entropy'] = df['clean_text'].apply(shannon_entropy)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(8, 6))
    sns.histplot(df['entropy'], bins=30)
    plt.title("Entropy Distribution")
    plt.tight_layout()
    plt.savefig("results/dedup/entropy_histogram.png")
    print("[+] Saved entropy_histogram.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='word_count', y='entropy')
    plt.title("Entropy vs Word Count")
    plt.tight_layout()
    plt.savefig("results/dedup/entropy_vs_wordcount.png")
    print("[+] Saved entropy_vs_wordcount.png")

    # Step 3: Metadata-Aware (optional)
    print("[?] Run metadata-aware duplication analysis? (y/n): ", end="")
    if input().strip().lower() == 'y':
        cols = input("Enter metadata columns (comma separated): ").split(',')
        cols = [c.strip() for c in cols if c.strip() in df.columns and df[c].dtype == 'object']

        for col in cols:
            for val, group in df.groupby(col):
                plt.figure(figsize=(8, 6))
                sns.countplot(data=group, x='dup_type')
                plt.title(f"Duplication Types for {col} = {val}")
                plt.tight_layout()
                fname = f"results/dedup/dup_type_{col}_{str(val).replace(' ', '_')}.png"
                plt.savefig(fname)
                print(f"[+] Saved {fname}")

    print("[✓] Deduplication pipeline complete. Ready for next command.")
