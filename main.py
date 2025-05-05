from app.file_loader import popup_csv_loader, trex_start
from app.embedding import generate_embeddings
import pandas as pd
from app.pipeline import metadata_viz
from app.dedup import trex_dedup
from app.clustering import run_clustering_pipeline
from app.text2sql_pipeline import run_text2sql_pipeline
import os


if __name__ == '__main__':
    print("Launching T.REX file loader...")
    df = popup_csv_loader()
    if df is not None:
        print("[+] File Loaded.")
        print(f"[+] Sample rows:\n{df.head(10)}")
        loaded_df, text_col = trex_start(df)
        if loaded_df is not None:
            print(f"[+] Metadata and text columns detected...\n{loaded_df.head(10)}")

            _, elapsed = generate_embeddings(loaded_df, text_col)
            print("[*] TREX Ready for Refinement and EXploration.")
            print("[*] Type a command (e.g., trex_taskName(param=['param1', 'param2']))")

            cached_df = None

            while True:
                cmd = input("T.REX > ").strip()

                if cmd.startswith("trex_eda"):
                    try:
                        exec_env = {}
                        exec(f"args_dict = dict{cmd[len('trex_eda'):].strip()}", {}, exec_env)
                        metadata_cols = exec_env.get("args_dict", {}).get("metadata", [])

                        if not metadata_cols:
                            print("⚠️  Please pass metadata column names. Example: trex_eda(metadata=['col1', 'col2'])")
                            continue

                        eda_df = cached_df if cached_df is not None else pd.read_csv("data/embedding_metadata.csv")
                        metadata_viz(eda_df, text_col, metadata_cols)
                        print("[*] EDA completed. Ready for next command.")

                    except Exception as e:
                        print(f"[ERROR] Failed to run EDA: {e}")

                elif cmd == "trex_restart()":
                    try:
                        cached_df = pd.read_csv("data/embedding_metadata.csv")
                        print("[+] Cached file reloaded into memory. You may rerun commands now.")
                    except Exception as e:
                        print(f"[ERROR] Failed to load cached file: {e}")
                
                elif cmd.startswith("trex_dedup"):
                    try:
                        exec_env = {}
                        exec(f"args_dict = dict{cmd[len('trex_dedup'):].strip()}", {}, exec_env)
                        stop_flag = exec_env.get("args_dict", {}).get("stopwords", False)

                        if stop_flag:
                            use_default = input("Use default stopwords list? (y/n): ").strip().lower()
                            if use_default == "y":
                                stopwords_list = None  # Use nltk default inside the pipeline
                            else:
                                custom = input("Enter custom stopwords (comma-separated): ").strip()
                                stopwords_list = [w.strip().lower() for w in custom.split(",") if w.strip()]
                        else:
                            stopwords_list = []

                        dedup_df = cached_df if cached_df is not None else pd.read_csv("data/embedding_metadata.csv")
                        trex_dedup(dedup_df, text_col, stopwords_list)
                        print("[*] Duplicate anlaysis complete. Ready for next command.")
                    
                    except Exception as e:
                        print(f"[ERROR] Failed to run Deduplication pipeline: {e}")

                elif cmd == "trex_cluster()":
                    try:
                        # Ensure data exists
                        embedding_path = "data/initial_embeddings.npy"
                        metadata_path = "data/embedding_metadata.csv"

                        if not os.path.exists(embedding_path) or not os.path.exists(metadata_path):
                            print("[!] Missing embedding or metadata file. Relaunch and load the file from start.")
                            continue

                        print("[*] Running clustering pipeline...")
                        run_clustering_pipeline(embedding_path, metadata_path)
                        print("[*] Clustering complete. Ready for next command.")

                    except Exception as e:
                        print(f"[ERROR] Failed to run clustering pipeline: {e}")
                
                elif cmd.startswith("trex_text2sql"):
                    try:
                        exec_env = {}
                        exec(f"args_dict = dict{cmd[len('trex_text2sql'):].strip()}", {}, exec_env)
                        pii_masking = exec_env.get("args_dict", {}).get("pii_mask", False)
                        text2sql_df = cached_df if cached_df is not None else pd.read_csv("data/embedding_metadata.csv")
                        run_text2sql_pipeline(text2sql_df, text_col, pii_mask=pii_masking)
                    except Exception as e:
                        print(f"[ERROR] Failed to run Text2SQL pipeline: {e}")

                elif cmd in ["exit", "quit"]:
                    print("Exiting T.REX.")
                    break

                else:
                    print("Unknown command. Try again.")
