import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import httpx
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STOPWORDS = set(stopwords.words("english"))

def run_elbow_plot(embeddings):
    inertias = []
    for k in range(2, 21):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), inertias, marker='o')
    plt.title("Elbow Plot for KMeans Clustering")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

def get_keywords(texts, topn=10):
    words = " ".join(texts)
    tokens = [w.lower() for w in word_tokenize(words) if w.isalpha() and w.lower() not in STOPWORDS]
    freq = pd.Series(tokens).value_counts()
    return freq.head(topn).index.tolist()

def label_clusters_with_llm(cluster_keywords, temperature=0.3, max_tokens=40):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    labels = []
    for i, kw in enumerate(cluster_keywords):
        prompt = (
            f"I will give you a list of keywords from a text cluster with stopwords already removed. Start your response with a short label (1–4 words) followed by ---"
            f"in JSON format like {{\"label\": \"Cluster Name\"}}.\n\nKeywords:\n{', '.join(kw)}"
        )
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You generate short cluster labels from keywords."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": '---'
        }

        for attempt in range(3):
            try:
                res = httpx.post(url, headers=headers, json=payload, timeout=10)
                if res.status_code == 200:
                    content = res.json()['choices'][0]['message']['content']
                    try:
                        label = json.loads(content.strip())["label"]
                        labels.append(label)
                        break
                    except:
                        match = re.search(r'"label"\s*:\s*"(.+?)"', content)
                        if match:
                            labels.append(match.group(1))
                            break
                time.sleep(1)
            except Exception as e:
                print(f"[!] Retry {attempt+1} failed:", e)
                time.sleep(1)
        else:
            labels.append("Sorry! Failed to generate label")

    return labels

def run_clustering_pipeline(embedding_path="data/initial_embeddings.npy", metadata_path="data/embedding_metadata.csv"):
    os.makedirs("results/clustering", exist_ok=True)

    # Load embeddings + metadata
    X = np.load(embedding_path)
    df = pd.read_csv(metadata_path)
    print(f"[+] Loaded {len(X)} embeddings.")

    # Elbow plot
    run_elbow_plot(X)
    try:
        k = int(input("Enter the number of clusters: "))
    except:
        print("Invalid input. Exiting.")
        return

    # Cluster and assign labels
    model = MiniBatchKMeans(n_clusters=k, batch_size=256, random_state=42)
    df["cluster"] = model.fit_predict(X)

    score = silhouette_score(X, df["cluster"])
    print(f"[+] Clustering complete. Silhouette Score: {score:.4f}")

    # Show wordclouds
    for i in range(k):
        texts = df[df["cluster"] == i]["text"].astype(str)
        keywords = get_keywords(texts)
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud - Cluster {i}")
        plt.tight_layout()
        plt.show()

        if input(f"Save wordcloud for cluster {i}? (y/n): ").strip().lower() == "y":
            wc.to_file(f"results/clustering/wordcloud_cluster_{i}.png")
            print(f"[+] Saved: wordcloud_cluster_{i}.png")

    # Ask to label using LLM
    if input("Do you want to auto-label using LLM? (y/n): ").strip().lower() == "y":
        print("[*] Sending cluster keywords to Groq API...")
        cluster_kw = [get_keywords(df[df["cluster"] == i]["text"].astype(str)) for i in range(k)]
        cluster_labels = label_clusters_with_llm(cluster_kw)

        print("\n=== Cluster Labels ===")
        for i, label in enumerate(cluster_labels):
            print(f"Cluster {i}: {label}")

    # Save results
    df.to_csv("data/clustered.csv", index=False)
    print("[+] Saved clustered dataset to data/clustered.csv")
