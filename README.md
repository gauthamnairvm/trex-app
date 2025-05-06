# T.REX — Text Refinement and EXploration

T.REX is a powerful local tool for analyzing, deduplicating, clustering, and querying large-scale text datasets using pretrained embeddings and LLM-powered pipelines with metadata aware plots and integrations.

> ❗ Requires a machine with a **dedicated NVIDIA GPU (CUDA 11.8+)**.  
> ❌ Will NOT run in Docker, WSL, or headless environments due to GUI popups.

---

## 🛠 Features

- ✅ CSV popup loader with column detection
- ✅ Embedding generation (`sentence-transformers`)
- ✅ Interactive EDA on metadata + selected text column
- ✅ Clustering and optional LLM labeling
- ✅ Near duplicate analysis
- ✅ Text-to-SQL pipeline
- ✅ Full CLI-based UX for pipeline chaining

---

## ⚙️ Installation

### 1. Prerequisites

- Python **3.10**
- NVIDIA GPU with **CUDA 11.8+ drivers installed**
- **Display environment** (no WSL or remote/headless)

### 2. Setup Steps

```bash
# Clone the repo
git clone https://github.com/gauthamnairvm/trex-app.git
cd trex-app

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

#Additional Dependencies
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

```

---

## 🔐 Environment Setup

Create a `.env` file at the root (you can copy from `.env.template`):

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Run T.REX

```bash
python main.py
```

You'll see the CSV loader and the T.REX CLI. Try:

```bash
T.REX > trex_eda(metadata=['col1', 'col2'])
T.REX > trex_cluster()
T.REX > trex_dedup(stopwords=False)
T.REX > trex_text2sql(pii_mask=True)
```

---

## 📂 Project Structure

```
trex-app/
├── app/
│   ├── clustering.py
│   ├── dedup.py
│   ├── embedding.py
│   ├── file_loader.py
│   ├── pipeline.py
│   └── text2sql_pipeline.py
├── data/               # CSVs and embeddings
├── results/            # Output plots and clustering results
├── main.py             # Entry point
├── .env.template       # Environment variable example
├── requirements.txt
└── README.md
```

---

## 📋 License

This project is licensed under the **MIT License**.  
You’re free to use, modify, and distribute it. Please give credit if you build on it.

---

## 🤝 Contributions

We welcome issues, suggestions, and pull requests.  
To contribute:

1. Fork the repo
2. Create a feature branch
3. Submit a PR with proper description

---

Built and maintained by `Variath Madhupal Gautham Nair (MSCS Rutgers University-New Brunswick)`