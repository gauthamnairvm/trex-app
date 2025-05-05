# T.REX: Text Refinement and EXploration Toolkit

T.REX is an interactive command-line tool designed to simplify text exploration pipelines using real-world NLP techniques and LLM-enhanced workflows.

## 🔧 Features
- Embedding generation and metadata enrichment
- Text-based EDA with interactive plots
- Deduplication analysis using MinHash, Shannon entropy & n-gram overlap
- Clustering with MiniBatchKMeans + LLM-based labeling
- Text-to-SQL powered by LLaMA3 via Groq API
- CLI-style prompts with popups and intermediate saves

## 🐳 Docker Quickstart

### 1. Build Docker Image
```bash
docker build -t trex-app .
```

### 2. Run Docker Container
```bash
docker run -it --rm   -v $(pwd)/data:/app/data   -v $(pwd)/results:/app/results   --env-file .env   trex-app
```

## 📁 Expected Folder Structure

```
TREX/
│
├── app/
│   ├── file_loader.py
│   ├── embedding.py
│   ├── pipeline.py
│   ├── dedup.py
│   └── config.py
│
├── data/
│   ├── entry_file.csv
│   ├── initial_embeddings.npy
│   ├── embedding_metadata.csv
│   └── embedding_row_ids.csv
│
├── results/
│   ├── eda/
│   ├── clustering/
│   └── dedup/
│
├── .env.template
├── .gitignore
├── Dockerfile
├── main.py
└── requirements.txt
```

## 🔐 Environment Variables

Copy `.env.template` to `.env` and set your Groq API key:

```
cp .env.template .env
```

Edit `.env`:
```
GROQ_API_KEY=your_actual_groq_key_here
```

---

© T.REX Project — 2025. Licensed under MIT. Free to use with credit to original developer - `Variath Madhupal Gautham Nair - MSCS Rutgers Univeristy` if building upon it.
