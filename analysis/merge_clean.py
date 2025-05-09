import pandas as pd
from pathlib import Path

def merge_clean(
    data_dir: str | Path,
    csv_pattern: str = "group*.csv",
    out_path: str | Path = "output/merged_cleaned.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge CSVs, standardize columns via an alias map, keep canonical fields only,
    drop rows without DOI & URL, and write the cleaned file.

    Parameters
    ----------
    data_dir     : folder that contains the raw CSV files
    csv_pattern  : glob pattern to select the CSVs (e.g. 'group*.csv')
    out_path     : file name to save the cleaned CSV
    verbose      : print progress if True

    Returns
    -------
    pd.DataFrame : the cleaned DataFrame
    """

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob(csv_pattern))
    if verbose:
        print(f"📂 Found {len(csv_files)} files in {data_dir}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    df.columns = df.columns.str.strip().str.lower()

    ALIASES = {
        "doi":  ["doi"],
        "url":  ["url"],
        "projectid":         ["project_id", "projectid"],
        "projectstatus":     ["project_status", "projectstatus"],
        "projecttitle":      ["projecttitle"],
        "projectrdc":        ["agency", "location", "rdc", "projectrdc"],
        "projectyearstarted": ["start year", "projectstartyear"],
        "projectyearended":   ["end year", "projectendyear"],
        "projectpi":          ["project_pi", "pi", "projectpi"],
        "researcher":         ["author", "authors", "researcher", "researchers"],
        "outputtitle":   ["title", "outputtitle"],
        "outputbiblio":  ["outputbiblio"],
        "outputtype":    ["outputtype"],
        "outputstatus":  ["outputstatus"],
        "outputvenue":   ["outputvenue"],
        "outputyear":    ["year", "outputyear", "publication_year"],
        "outputmonth":   ["outputmonth"],
        "outputvolume":  ["outputvolume"],
        "outputnumber":  ["outputnumber"],
        "outputpages":   ["outputpages"],
    }

    rename_map = {
        alias.strip().lower(): canon
        for canon, lst in ALIASES.items()
        for alias in lst
    }
    df.rename(columns=rename_map, inplace=True)

    for canon, alt_list in ALIASES.items():
        canon_lc = canon.lower()
        present = [
            c for c in df.columns
            if c in [canon_lc] + [a.lower() for a in alt_list]
        ]
        if not present:
            continue
        df[canon_lc] = df[present].bfill(axis=1).iloc[:, 0]
        df.drop(columns=[c for c in present if c != canon_lc],
                inplace=True, errors="ignore")

    df = df.loc[:, ~df.columns.duplicated()]

    df.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "N/A": pd.NA}, inplace=True)

    df = df.reset_index(drop=True)
    for col in ["doi", "url"]:
        if col not in df.columns:
            df[col] = pd.NA
    df_clean = df.loc[~(df["doi"].isna() & df["url"].isna())].copy()

    canon_cols = list(ALIASES.keys()) + ["doi", "url"]
    df_clean = df_clean[[c for c in canon_cols if c in df_clean.columns]]
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output dir exists
    df_clean.to_csv(out_path, index=False, encoding="utf-8")
    if verbose:
        print(f"✅ Cleaned file saved to {out_path}")

    return df_clean
