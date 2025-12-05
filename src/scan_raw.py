import os
import glob
try:
    import pandas as pd  # optional
except Exception:
    pd = None
import csv

def find_dataset_files(root: str = "data/raw") -> list[str]:
    pattern = os.path.join(root, "**", "*Dataset_*.csv")
    return glob.glob(pattern, recursive=True)

def parse_indicator(dir_name: str) -> tuple[str, str]:
    if "_" in dir_name:
        parts = dir_name.split("_", 1)
        return parts[0], parts[1]
    return dir_name, ""

def summarize_dataset(path: str) -> dict:
    dir_name = os.path.basename(os.path.dirname(path))
    indicator_id, indicator_desc = parse_indicator(dir_name)
    if pd is not None:
        try:
            df = pd.read_csv(path)
            rows, cols = df.shape
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
            categorical = [c for c in df.columns if c not in numeric]
            total = rows * cols if rows * cols > 0 else 1
            missing_rate = float(df.isna().sum().sum()) / float(total)
            return {
                "file": path,
                "indicator_id": indicator_id,
                "indicator_desc": indicator_desc,
                "rows": rows,
                "cols": cols,
                "numeric_cols": len(numeric),
                "categorical_cols": len(categorical),
                "missing_rate": missing_rate,
            }
        except Exception as e:
            return {"file": path, "indicator_id": indicator_id, "indicator_desc": indicator_desc, "error": str(e)}
    # csv fallback
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = len(header)
            rows = 0
            missing_cells = 0
            numeric_flags = [True] * cols
            for row in reader:
                rows += 1
                for j in range(cols):
                    cell = row[j] if j < len(row) else ""
                    if str(cell).strip() == "":
                        missing_cells += 1
                    else:
                        try:
                            float(cell)
                        except Exception:
                            numeric_flags[j] = False
            total = rows * cols if rows * cols > 0 else 1
            missing_rate = float(missing_cells) / float(total)
            numeric_cols = sum(1 for flag in numeric_flags if flag)
            categorical_cols = cols - numeric_cols
            return {
                "file": path,
                "indicator_id": indicator_id,
                "indicator_desc": indicator_desc,
                "rows": rows,
                "cols": cols,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "missing_rate": missing_rate,
            }
    except Exception as e:
        return {"file": path, "indicator_id": indicator_id, "indicator_desc": indicator_desc, "error": str(e)}

def build_summary(root: str = "data/raw", out_path: str = "data/processed/raw_summary.csv") -> str:
    files = find_dataset_files(root)
    records = [summarize_dataset(p) for p in files]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if pd is not None:
        pd.DataFrame(records).to_csv(out_path, index=False)
    else:
        if records:
            fieldnames = list(records[0].keys())
        else:
            fieldnames = ["file", "indicator_id", "indicator_desc", "rows", "cols", "numeric_cols", "categorical_cols", "missing_rate", "error"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
    return out_path

def main():
    path = build_summary()
    print(path)

if __name__ == "__main__":
    main()
