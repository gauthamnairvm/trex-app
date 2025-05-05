import os
import httpx
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from dotenv import load_dotenv
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = set(stopwords.words("english"))

def mask_pii(text):
    if pd.isnull(text): return ""
    text = str(text)
    text = re.sub(r"\b[\w.-]+?@\w+?\.\w+?\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{10,}\b|\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE]", text)
    text = re.sub(r"\b\d+\b", "[NUMBER]", text)
    return text

def run_text2sql_pipeline(df, text_col, pii_mask=False):
    print("[*] Text2SQL Pipeline Started")
    print(f"[+] Main text column: {text_col}")

    meta_cols = [col for col in df.columns if col != text_col]
    print("[+] Available metadata columns:")
    for i, col in enumerate(meta_cols):
        print(f"  {i+1}. {col}")

    raw = input("Enter comma-separated column names to include in context (or press Enter for none): ").strip()
    selected_cols = [text_col] + [c.strip() for c in raw.split(",") if c.strip() in meta_cols]

    embed_matrix = np.load("data/initial_embeddings.npy")
    print(f"[+] Loaded embeddings with shape: {embed_matrix.shape}")

    while True:
        user_question = input("\nAsk a question (or type 'exit' to stop): ").strip()
        if user_question.lower() == "exit":
            print("Exiting Text2SQL.")
            break

        query_embed = MODEL.encode([user_question])
        similarities = cosine_similarity(query_embed, embed_matrix)[0]
        top_idxs = np.argsort(similarities)[-5:][::-1]

        context_rows = []
        for i in top_idxs:
            row_data = []
            for col in selected_cols:
                val = df.iloc[i][col]
                if pii_mask:
                    val = mask_pii(val)
                row_data.append(f"{col}: {val}")
            context_rows.append(f"[Row {i}] " + "; ".join(row_data))

        context = "\n\n".join(context_rows)

        system_prompt = (
            "You are a SQL expert helping generate SQLite queries. "
            f"Use ONLY the following columns: {', '.join(df.columns)}"
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"User question:\n{user_question}\n\n"
            "Write a valid SQL query to answer the question using the table 'df'."
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }

        try:
            response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"].strip()
                print("\nGenerated SQL Query:\n", reply)
                user_query = input("Press Enter to run this, or paste your own SQL query: ").strip() or reply

                conn = sqlite3.connect(":memory:")
                df.to_sql("df", conn, index=False)
                try:
                    result = pd.read_sql_query(user_query, conn)
                    print("\n[+] Query result:")
                    print(result.head())
                except Exception as e:
                    print("[!] Query execution failed:", e)
                finally:
                    conn.close()
            else:
                print(f"[!] LLM Error: {response.status_code} - {response.text}")
        except Exception as e:
            print("[!] HTTP error:", e)
