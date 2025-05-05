import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


SUPPORTED_TYPES = ['str', 'int', 'float', 'bool', 'category']

def popup_csv_loader():
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(title="Select your CSV file", filetypes=[("CSV files", "*.csv")])
    if not filepath:
        messagebox.showerror("Error", "No file selected.")
        return None

    if not filepath.lower().endswith(".csv"):
        messagebox.showerror("Unsupported File Type", "Only CSV files are supported.")
        return None

    while True:
        try:
            # Get CSV settings
            encoding = simpledialog.askstring("Encoding", "Enter encoding (default=utf-8):", initialvalue="utf-8")
            if not encoding:
                encoding = 'utf-8'

            delimiter = simpledialog.askstring("Delimiter", "Enter delimiter (default=,):", initialvalue=",")
            if not delimiter:
                delimiter = ','

            use_header = messagebox.askyesno("Header Row", "Does the file have a header row?")

            if not use_header:
                header = None
                columns_str = simpledialog.askstring("Column Names", "Enter comma-separated column names:")
                if not columns_str:
                    raise ValueError("You must provide column names if no header row is used.")
                column_names = [x.strip() for x in columns_str.split(',')]
            else:
                header = 0
                column_names = None

            df = pd.read_csv(filepath, encoding=encoding, delimiter=delimiter, header=header)

            if column_names:
                df.columns = column_names

            os.makedirs("data", exist_ok=True)
            save_path = os.path.join("data", "entry_file.csv")
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Accepted file type.")
            return df

        except Exception as e:
            messagebox.showerror("Load Failed", f"{str(e)}\n\nPlease enter valid inputs.")
            continue

def trex_start(input_df):
    import traceback

    df_holder = {"df": None}
    text_column_holder = {"name": None}
    df = input_df.copy()
    cast_types = {}

    def on_confirm():
        try:
            new_columns = [entry.get() for entry in rename_entries]
            df.columns = new_columns

            selected_text = text_col.get()
            text_column_holder["name"] = selected_text
            selected_meta = [meta_listbox.get(i) for i in meta_listbox.curselection()]
            if not selected_text or not selected_meta:
                messagebox.showerror("Missing Selection", "Please select a text column and at least one metadata column.")
                return

            all_selected = [selected_text] + selected_meta
            df_filtered = df[all_selected].copy()

            for col in selected_meta:
                cast_type = cast_types[col].get()
                if cast_type:
                    df_filtered[col] = df_filtered[col].astype(cast_type)

            df_filtered.insert(0, 'row_id', df_filtered.index)
            df_filtered.to_csv("data/trex_start.csv", index=False)
            messagebox.showinfo("Success", "File loaded.")
            df_holder["df"] = df_filtered
            popup.quit()
            popup.destroy()

        except Exception as e:
            print("[ERROR] Exception during file save:")
            traceback.print_exc()
            messagebox.showerror("Save Error", str(e))

    popup = tk.Tk()
    popup.title("T.REX Column Setup")
    popup.geometry("700x600")

    canvas = tk.Canvas(popup)
    scrollbar = tk.Scrollbar(popup, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    columns = list(df.columns)
    rename_entries = []

    tk.Label(scroll_frame, text="Rename Columns").pack()
    for col in columns:
        frame = tk.Frame(scroll_frame)
        frame.pack()
        tk.Label(frame, text=col, width=20, anchor='w').pack(side='left')
        entry = tk.Entry(frame)
        entry.insert(0, col)
        entry.pack(side='left')
        rename_entries.append(entry)

    tk.Label(scroll_frame, text="\nSelect Text Column").pack()
    text_col = ttk.Combobox(scroll_frame, values=columns)
    text_col.pack()

    tk.Label(scroll_frame, text="\nSelect Metadata Columns").pack()
    meta_listbox = tk.Listbox(scroll_frame, selectmode=tk.MULTIPLE, height=6)
    for col in columns:
        meta_listbox.insert(tk.END, col)
    meta_listbox.pack()

    tk.Label(scroll_frame, text="\nType Cast Metadata Columns").pack()
    for col in columns:
        row = tk.Frame(scroll_frame)
        row.pack()
        tk.Label(row, text=col, width=20, anchor='w').pack(side='left')
        var = tk.StringVar()
        cast_menu = ttk.Combobox(row, textvariable=var, values=SUPPORTED_TYPES, width=10)
        cast_menu.pack(side='left')
        cast_types[col] = var

    tk.Button(scroll_frame, text="OK", command=on_confirm).pack(pady=10)
    popup.mainloop()

    return df_holder["df"],  text_column_holder["name"]

