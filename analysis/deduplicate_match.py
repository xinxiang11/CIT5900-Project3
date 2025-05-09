import pandas as pd
import re
from collections import Counter
from rapidfuzz import process, fuzz

def process_and_match_publications(
    csv_path="output/merged_cleaned.csv",
    excel_path="data/ProjectsAllMetadata.xlsx",
    output_csv_path="output/matched_publications.csv",
    verbose=True
) -> pd.DataFrame:
    # Step 1: Load publication data
    df = pd.read_csv(csv_path, low_memory=False)

    # Step 2: Clean DOI and Title
    def clean_doi(doi):
        if not isinstance(doi, str):
            return None
        doi = doi.strip().lower()
        if "doi.org/" in doi:
            return doi.split("doi.org/")[-1]
        return doi

    def clean_title(title):
        if not isinstance(title, str):
            return None
        title = title.lower()
        title = re.sub(r'\s+', '', title)
        return title

    df["clean_doi"] = df["doi"].apply(clean_doi)
    df["clean_title"] = df["outputtitle"].apply(clean_title)

    # Step 3: Deduplicate by DOI and Title
    def keep_first_nonmissing(group):
        return group.apply(lambda col: col.dropna().iloc[0] if col.dropna().any() else None)

    df_doi_deduped = df[df["clean_doi"].notna()].groupby("clean_doi", group_keys=False).apply(keep_first_nonmissing).reset_index(drop=True)
    df_title_deduped = df[df["clean_doi"].isna()].groupby("clean_title", group_keys=False).apply(keep_first_nonmissing).reset_index(drop=True)
    df_deduplicated = pd.concat([df_doi_deduped, df_title_deduped], ignore_index=True)
    df_deduplicated.drop(columns=["clean_doi", "clean_title"], inplace=True)

    # Step 4: Load metadata and normalize
    meta_df = pd.read_excel(excel_path, sheet_name="All Metadata")
    researcher_df = pd.read_excel(excel_path, sheet_name="Researchers")

    df_deduplicated.columns = df_deduplicated.columns.str.strip().str.lower()
    meta_df.columns = meta_df.columns.str.strip()
    researcher_df.columns = researcher_df.columns.str.strip()

    meta_df["norm_pi"] = meta_df["PI"].astype(str).str.strip().str.lower()
    meta_df["norm_title"] = meta_df["Title"].astype(str).str.strip()
    researcher_df["norm_researcher"] = researcher_df["Researcher"].astype(str).str.strip().str.lower()

    # Step 5: Build lookup dictionaries
    pi_to_projects = meta_df.groupby("norm_pi")[["Proj ID", "Title"]].apply(lambda df: df.to_dict("records")).to_dict()
    res_to_projects = researcher_df.groupby("norm_researcher")[["Proj ID", "Title"]].apply(lambda df: df.to_dict("records")).to_dict()

    # Step 6: Matching logic
    matched_projids = []
    matched_sources = []

    for _, row in df_deduplicated.iterrows():
        title = str(row.get("outputtitle", "")).strip()
        matched = False

        # Try PI match
        pi = str(row.get("projectpi", "")).strip().lower()
        if pi and pi in pi_to_projects:
            projects = pi_to_projects[pi]
            if len(projects) == 1:
                proj_title = projects[0]["Title"]
                score = fuzz.token_sort_ratio(title, proj_title)
                if score >= 40:
                    matched_projids.append(projects[0]["Proj ID"])
                    matched_sources.append("unique_pi_match")
                    continue
            else:
                titles = [p["Title"] for p in projects]
                match, score, idx = process.extractOne(title, titles, scorer=fuzz.token_sort_ratio)
                if score >= 60:
                    matched_projids.append(projects[idx]["Proj ID"])
                    matched_sources.append("fuzzy_pi_title_match")
                    continue

        # Try researcher match
        val = row.get("researcher", None)
        if pd.notna(val):
            candidates = str(val).replace(",", ";").split(";")
            for name in candidates:
                name_clean = name.strip().lower()
                if name_clean in res_to_projects:
                    projects = res_to_projects[name_clean]
                    if len(projects) == 1:
                        proj_title = projects[0]["Title"]
                        score = fuzz.token_sort_ratio(title, proj_title)
                        if score >= 40:
                            matched_projids.append(projects[0]["Proj ID"])
                            matched_sources.append("unique_researcher_match")
                            matched = True
                            break
                    else:
                        titles = [p["Title"] for p in projects]
                        match, score, idx = process.extractOne(title, titles, scorer=fuzz.token_sort_ratio)
                        if score >= 60:
                            matched_projids.append(projects[idx]["Proj ID"])
                            matched_sources.append("fuzzy_researcher_title_match")
                            matched = True
                            break
            if matched:
                continue

        # No match
        matched_projids.append(None)
        matched_sources.append(None)

    # Step 7: Append matching results
    df_deduplicated["matchedprojid"] = matched_projids
    df_deduplicated["matchedsource"] = matched_sources

    # Step 8: Filter to matched only
    df_matched = df_deduplicated[df_deduplicated["matchedprojid"].notna()].reset_index(drop=True)

    # Step 9: Load metadata dictionaries for filling
    meta_dict = meta_df.set_index("Proj ID")[["PI", "Status", "RDC", "Start Year", "End Year", "Title"]].to_dict("index")
    res_dict = researcher_df.groupby("Proj ID")["Researcher"].apply(lambda x: "; ".join(x)).to_dict()

    # Step 10: Fill missing fields
    for i, row in df_matched.iterrows():
        projid = row.get("matchedprojid")
        if pd.isna(projid):
            continue

        if pd.isna(row.get("projectid")):
            df_matched.at[i, "projectid"] = projid

        meta_row = meta_dict.get(projid)
        if meta_row:
            if pd.isna(row.get("projectpi")) and pd.notna(meta_row.get("PI")):
                df_matched.at[i, "projectpi"] = meta_row["PI"]
            if pd.isna(row.get("projectstatus")) and pd.notna(meta_row.get("Status")):
                df_matched.at[i, "projectstatus"] = meta_row["Status"]
            if pd.isna(row.get("projectrdc")) and pd.notna(meta_row.get("RDC")):
                df_matched.at[i, "projectrdc"] = meta_row["RDC"]
            if pd.isna(row.get("projectyearstarted")) and pd.notna(meta_row.get("Start Year")):
                df_matched.at[i, "projectyearstarted"] = meta_row["Start Year"]
            if pd.isna(row.get("projectyearended")) and pd.notna(meta_row.get("End Year")):
                df_matched.at[i, "projectyearended"] = meta_row["End Year"]
            if pd.isna(row.get("projecttitle")) and pd.notna(meta_row.get("Title")):
                df_matched.at[i, "projecttitle"] = meta_row["Title"]

        if pd.isna(row.get("researcher")) and projid in res_dict:
            df_matched.at[i, "researcher"] = res_dict[projid]

    # Step 11: Drop intermediate columns
    df_matched.drop(columns=["matchedprojid", "matchedsource"], inplace=True)

    # Step 12: Save final output
    df_matched.to_csv(output_csv_path, index=False)
    if verbose:
        print(f"✅ Matched file saved to {output_csv_path}")

    return df_matched
