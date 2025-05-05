import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(df, text_col):
    start_time = time.time()

    texts = df[text_col].astype(str).tolist()
    ids = df['row_id'].tolist()

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        device='cuda',
        num_workers=6,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    os.makedirs("data", exist_ok=True)
    np.save("data/initial_embeddings.npy", embeddings)

    rowid_df = pd.DataFrame({'row_id': ids})
    rowid_df.to_csv("data/embedding_row_ids.csv", index=False)

    # Compute stats
    embed_mean = embeddings.mean(axis=1)
    embed_std = embeddings.std(axis=1)
    embed_norm = np.linalg.norm(embeddings, axis=1)

    word_count = df[text_col].astype(str).apply(lambda x: len(x.split()))
    char_count = df[text_col].astype(str).apply(len)

    stats_df = df.copy()
    stats_df['embed_mean'] = embed_mean
    stats_df['embed_std'] = embed_std
    stats_df['embed_norm'] = embed_norm
    stats_df['word_count'] = word_count
    stats_df['char_count'] = char_count

    stats_df.to_csv("data/embedding_metadata.csv", index=False)

    elapsed = time.time() - start_time
    print(f"[+] Embeddings generated and cached in {elapsed:.2f} seconds")

    return embeddings, elapsed
