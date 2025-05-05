import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.express as px
import pandas as pd
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ALLOWED_OBJECT_TYPES = ['object', 'category', 'bool']
ALLOWED_NUMERIC_TYPES = ['int64', 'float64']


def metadata_viz(df, text_col, columns):
    os.makedirs("results/eda", exist_ok=True)

    # 1. Distribution Plot
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        if df[col].dtype.name in ALLOWED_OBJECT_TYPES:
            df[col].value_counts().plot(kind='bar', ax=ax, title=f"Distribution of {col}")
        elif df[col].dtype.name in ALLOWED_NUMERIC_TYPES:
            sns.histplot(df[col], bins=30, ax=ax)
            ax.set_title(f"Distribution of {col}")
        else:
            ax.set_title(f"Skipped unsupported column: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    save = input("Save this distribution plot? (y/n): ").strip().lower()
    if save == 'y':
        plt.savefig("results/eda/metadata_distributions.png")
        print("[+] Saved metadata_distributions.png")

    # 2. Correlation heatmap (only for numeric cols)
    corr_cols = [c for c in columns if df[c].dtype.name in ALLOWED_NUMERIC_TYPES] + ['embed_mean', 'embed_std', 'embed_norm', 'word_count', 'char_count']
    corr_data = df[corr_cols].select_dtypes(include='number')
    if not corr_data.empty:
        corr_matrix = corr_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        save = input("Save correlation heatmap? (y/n): ").strip().lower()
        if save == 'y':
            plt.savefig("results/eda/correlation_heatmap.png")
            print("[+] Saved correlation_heatmap.png")

    # 3. Interactive Plotly: word_count vs char_count
    hover_cols = [text_col, 'word_count', 'char_count'] + columns
    fig = px.scatter(
        df,
        x="word_count",
        y="char_count",
        hover_data=hover_cols,
        title="Text Distribution: Word Count vs Character Count",
        color_discrete_sequence=['#636EFA'],
        render_mode='webgl'
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    fig.show()

    save = input("Save this interactive plot (HTML)? (y/n): ").strip().lower()
    if save == 'y':
        fig.write_html("results/eda/text_word_char_distribution.html")
        print("[+] Saved text_word_char_distribution.html")

    # 4. Embedding mean vs char count with outliers (Plotly version)
    df['outlier'] = df['embed_mean'] > df['embed_mean'].quantile(0.95)
    hover_cols = [text_col, 'embed_mean', 'char_count'] + columns
    fig = px.scatter(
        df,
        x='embed_mean',
        y='char_count',
        color='outlier',
        hover_data=hover_cols,
        title="Embedding Mean vs Char Count (Outliers Highlighted)",
        color_discrete_map={True: 'red', False: 'blue'}
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    fig.show()

    save = input("Save outlier plot (HTML)? (y/n): ").strip().lower()
    if save == 'y':
        fig.write_html("results/eda/embedding_vs_charcount_outliers.html")
        print("[+] Saved embedding_vs_charcount_outliers.html")

    # 5. Stopword, Junk, Emoji analysis per valid categorical metadata column
    sw_set = set(stopwords.words('english'))
    for col in columns:
        if df[col].dtype.name not in ALLOWED_OBJECT_TYPES or df[col].nunique() > 50:
            print(f"[!] Skipping token category plot for {col} due to high cardinality or unsupported dtype.")
            continue

        grouped = df.groupby(col)
        for group_val, sub_df in grouped:
            stopword_counts, junk_counts, emoji_counts, other_counts = [], [], [], []
            for text in sub_df[text_col].astype(str):
                tokens = word_tokenize(text)
                stop_ct = sum(1 for t in tokens if t.lower() in sw_set)
                emoji_ct = sum(1 for c in text if emoji.is_emoji(c))
                junk_ct = len(re.findall(r"[^\w\s]", text)) - emoji_ct
                total = len(tokens)
                stopword_counts.append(stop_ct)
                junk_counts.append(junk_ct)
                emoji_counts.append(emoji_ct)
                other_counts.append(max(total - stop_ct - junk_ct - emoji_ct, 0))

            plt.figure(figsize=(10, 6))
            plt.hist([stopword_counts, junk_counts, emoji_counts, other_counts], bins=30, label=['Stopwords', 'Junk', 'Emoji', 'Other'])
            plt.title(f"Token Category Distribution for {col} = {group_val}")
            plt.xlabel("Count")
            plt.ylabel("Number of Rows")
            plt.legend()
            plt.tight_layout()
            plt.show()

            save = input(f"Save token category plot for {col}={group_val}? (y/n): ").strip().lower()
            if save == 'y':
                fname = f"results/eda/token_categories_{col}_{str(group_val).replace(' ', '_')}.png"
                plt.savefig(fname)
                print(f"[+] Saved {fname}")

    # 6. Token count distribution
    token_lens = df[text_col].astype(str).apply(lambda x: len(word_tokenize(x)))
    plt.figure(figsize=(8,6))
    sns.histplot(token_lens, bins=40)
    plt.title("Token Count Distribution")
    plt.xlabel("Tokens per Text")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    save = input("Save token count histogram? (y/n): ").strip().lower()
    if save == 'y':
        plt.savefig("results/eda/token_count_distribution.png")
        print("[+] Saved token_count_distribution.png")
